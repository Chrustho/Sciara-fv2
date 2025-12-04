#pragma once

#include "../../src/vent.h"
#include "../../src/Sciara.h"
#include "../../implementations/tiled/kernel_tiled.cuh"



__global__ void computeOutflows_Tiled(
    Sciara *sciara, const unsigned int tileX, const int tileY){
  // Parametri del dominio
  int rows= sciara->domain->rows;
  int cols= sciara->domain->cols;

  // Vicini
  int *xi= sciara->X->Xi;
  int *xj=sciara->X->Xj;

  // Buffers
  double *sh=sciara->substates->Sh;
  double *st=sciara->substates->ST;
  double *sz= sciara->substates->Sz;
  double *mf= sciara->substates->Mf;

  // a, b, c, d
  double _a= sciara->parameters->a;
  double _b=sciara->parameters->b;
  double _c=sciara->parameters->c;
  double _d=sciara->parameters->d;

  int blockSize = blockDim.x * blockDim.y;
  //taglia della shared dim(sz)*2 (sto considerando Sh,st)
  //rimane mf fuori

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
  int tid= tr* tileX +tc;

  if (i <rows && j < cols)
  {
    sh_s[tid] = sh[idx];
    st_s[tid] = st[idx];
    sz_s[tid] = sz[idx];


  } 
  __syncthreads();


  /* COMPUTAZIONE */
  double h0 = sh_s[idx];
  if (h0 <= 0.0) return;

  bool eliminated[MOORE_NEIGHBORS];
  double z[MOORE_NEIGHBORS];
  double h[MOORE_NEIGHBORS];
  double H[MOORE_NEIGHBORS];
  double theta[MOORE_NEIGHBORS];
  double Pr[MOORE_NEIGHBORS]; 
  double w[MOORE_NEIGHBORS];

  double sz0 = sz_s[idx];
  double T_val = st[idx]; 

  double rr = pow(10.0, _a + _b * T_val);
  double hc = pow(10.0, _c + _d * T_val);

  for (int k = 0; k < MOORE_NEIGHBORS; k++)
  {
    int ni = i + xi[k];
    int nj = j + xj[k];

    bool is_valid = (ni >= 0 && ni < rows && nj >= 0 && nj < cols);

    if (is_valid) {
      int idx_k = ni * cols + nj;
      double sz_k = sz[idx_k];
      h[k] = sh[idx_k];

      if (k < VON_NEUMANN_NEIGHBORS)
        z[k] = sz_k;
      else
        z[k] = sz0 - (sz0 - sz_k) / sqrt(2.0); 
    } 

    w[k] = pc; // Questo va impostato comunque
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
    Sciara *sciara);  


__global__ void computeNewTemperatureAndSolidification_Tiled(
    Sciara *sciara         
);

__global__ void boundaryConditions_Tiled(
    Sciara *sciara     
);

