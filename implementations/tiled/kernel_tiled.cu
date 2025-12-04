#pragma once

#include "../../src/vent.h"
#include "../../src/Sciara.h"
#include "../../implementations/tiled/kernel_tiled.cuh"

__constant__ int _Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1}; // Xj: Moore neighborhood row coordinates (see below)
__constant__ int _Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1}; // Xj: Moore neighborhood col coordinates (see below)


__global__ void computeOutflows_Tiled(
    Sciara *sciara){
  int rows= sciara->domain->rows;
  int cols= sciara->domain->cols;

  double *sh=sciara->substates->Sh;
  double *st=sciara->substates->ST;
  double *sz= sciara->substates->Sz;
  double *mf= sciara->substates->Mf;

  double _a= sciara->parameters->a;
  double _b=sciara->parameters->b;
  double _c=sciara->parameters->c;
  double _d=sciara->parameters->d;
  double pc=sciara->parameters->Pc;

  int blockSize = blockDim.x * blockDim.y;

  extern __shared__ double shared_mem[];

  double *sh_s = (double*)(shared_mem);
  double *st_s = (double*)(shared_mem + blockSize);
  double *sz_s = (double*)(shared_mem + blockSize * 2);

  int tc= threadIdx.x;
  int tr= threadIdx.y;

  int buffer_size= rows * cols *sizeof(double);

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int idx = i * cols + j;
  int tid= tr* blockDim.x +tc;

  if (i <rows && j < cols)
  {
    sh_s[tid] = sh[idx];
    st_s[tid] = st[idx];
    sz_s[tid] = sz[idx];
  }
  __syncthreads();

  if (i >= rows || j >= cols) return;


  /* COMPUTAZIONE */
  double h0 = sh_s[tid];
  if (h0 <= 0.0) return;

  bool eliminated[MOORE_NEIGHBORS];
  double z[MOORE_NEIGHBORS];
  double h[MOORE_NEIGHBORS];
  double H[MOORE_NEIGHBORS];
  double theta[MOORE_NEIGHBORS];
  double Pr[MOORE_NEIGHBORS]; 
  double w[MOORE_NEIGHBORS];

  double sz0 = sz_s[tid];
  double T_val = st_s[tid]; 

  double rr = pow(10.0, _a + _b * T_val);
  double hc = pow(10.0, _c + _d * T_val);

  for (int k = 0; k < MOORE_NEIGHBORS; k++)
  {
    int ni = i + _Xi[k];
    int nj = j + _Xj[k];

    int trn = tr + _Xi[k];
    int tcn = tc+ _Xj[k];

    bool is_valid = (ni >= 0 && ni < rows && nj >= 0 && nj < cols);

    if (is_valid) {
      int idx_k = ni * cols + nj;
      bool in_shared = (trn >= 0 && trn < (int)blockDim.y && 
                              tcn >= 0 && tcn < (int)blockDim.x);
            
      double sz_k;
      if (in_shared) {
          int idx_k_s = trn * blockDim.x + tcn;
          sz_k = sz_s[idx_k_s];
          h[k] = sh_s[idx_k_s];
      } else {
          sz_k = sz[idx_k];
          h[k] = sh[idx_k];
      }

      if (k < VON_NEUMANN_NEIGHBORS)
        z[k] = sz_k;
      else
        z[k] = sz0 - (sz0 - sz_k) / sqrt(2.0); 
    }

    w[k] = pc;
    Pr[k] = rr;
  }

  H[0] = z[0];
  theta[0] = 0.0;
  eliminated[0] = false;

  for (int k = 1; k < MOORE_NEIGHBORS; k++)
  {
    if (z[0] + h[0] > z[k] + h[k])
    {
      H[k] = z[k] + h[k];
      theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
      eliminated[k] = false;
    }
    else
    {
      eliminated[k] = true;
      H[k] = 0.0; 
      theta[k]=0.;
    }
  }

  bool loop;
  double avg;
  int counter;

  do
  {
    loop = false;
    avg = h[0];
    counter = 0;

    for (int k = 0; k < MOORE_NEIGHBORS; k++)
    {
      if (!eliminated[k])
      {
        avg += H[k];
        counter++;
      }
    }

    if (counter != 0)
      avg = avg / (double)counter;

    for (int k = 0; k < MOORE_NEIGHBORS; k++)
    {
      if (!eliminated[k] && avg <= H[k])
      {
        eliminated[k] = true;
        loop = true;
      }
    }
  } while (loop);

  for (int k = 1; k < MOORE_NEIGHBORS; k++)
  {
    int outflow_idx = k - 1; 
    int mf_idx = (outflow_idx * rows * cols) + idx;

    if (!eliminated[k] && h[0] > hc * cos(theta[k]))
    {
      mf[mf_idx] = Pr[k] * (avg - H[k]);
    }
    else
    {
      mf[mf_idx] = 0.0;
    }
  }

}


__global__ void massBalance_Tiled(
    Sciara *sciara) {

    int rows = sciara->domain->rows;
    int cols = sciara->domain->cols;

    double *sh = sciara->substates->Sh;
    double *sh_next = sciara->substates->Sh_next;
    double *st = sciara->substates->ST;
    double *st_next = sciara->substates->ST_next;
    double *mf = sciara->substates->Mf;

    int blockSize = blockDim.x * blockDim.y;

    extern __shared__ double shared_mem[];
    double *sh_s = shared_mem;
    double *st_s = shared_mem + blockSize;
    double *mf_s = shared_mem + blockSize * 2;

    int tc = threadIdx.x;
    int tr = threadIdx.y;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = i * cols + j;
    int tid = tr * blockDim.x + tc;

    int layer_size = rows * cols;

    
    if (i < rows && j < cols) {
        sh_s[tid] = sh[idx];
        st_s[tid] = st[idx];
        
        for (int layer = 0; layer < NUMBER_OF_OUTFLOWS; layer++) {
            mf_s[layer * blockSize + tid] = mf[layer * layer_size + idx];
        }
    }
    __syncthreads();


    if (i >= rows || j >= cols) return;

    const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};

    double initial_h = sh_s[tid];
    double initial_t = st_s[tid];

    double h_next = initial_h;
    double t_next = initial_h * initial_t;

    for (int n = 1; n < MOORE_NEIGHBORS; n++)
    {
        int ni = i + _Xi[n];
        int nj = j + _Xj[n];

        if (ni < 0 || ni >= rows || nj < 0 || nj >= cols)continue;

        int n_idx = ni * cols + nj;

        int out_layer = n - 1;
        double outFlow = mf_s[out_layer * blockSize + tid];

        int in_layer = inflowsIndices[n - 1];
        
        int trn = tr + _Xi[n];
        int tcn = tc + _Xj[n];
        
        bool in_shared = (trn >= 0 && trn < (int)blockDim.y && 
                          tcn >= 0 && tcn < (int)blockDim.x);

        double inFlow;
        double neigh_t;

        if (in_shared) {
            int tid_neighbor = trn * blockDim.x + tcn;
            inFlow = mf_s[in_layer * blockSize + tid_neighbor];
            neigh_t = st_s[tid_neighbor];
        } else {
            inFlow = mf[in_layer * layer_size + n_idx];
            neigh_t = st[n_idx];
        }

        h_next += (inFlow - outFlow);
        t_next += (inFlow * neigh_t - outFlow * initial_t);
    }

    if (h_next > 0) {
        t_next /= h_next;
        st_next[idx] = t_next;
        sh_next[idx] = h_next;
    }
}


__global__ void computeNewTemperatureAndSolidification_Tiled();
// In questo caso il kernel non beneficia del tiling perchè accede solo alla propria cella 
//e non a quella dei vicini

__global__ void boundaryConditions_Tiled();
// In questo caso il kernel non beneficia del tiling perchè accede solo alla propria cella 
//e non a quella dei vicini

