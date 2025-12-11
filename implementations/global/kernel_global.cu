#include "../../src/vent.h"
#include "../../src/Sciara.h"
#include "kernel_global.cuh"
#include "../../constants.cuh" 


__global__ void  computeOutflows_Global(
    double *sh, double *st, double *sz, double *mf)
{

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= rows || j >= cols) return;

  int idx = i * cols + j;

  double h0 = sh[idx];
  if (h0 <= 0.0) return;

  bool eliminated[MOORE_NEIGHBORS];
  double z[MOORE_NEIGHBORS];
  double h[MOORE_NEIGHBORS];
  double H[MOORE_NEIGHBORS];
  double theta[MOORE_NEIGHBORS];

  double sz0 = sz[idx];
  double T_val = st[idx]; 

  double rr = pow(10.0, d_a +d_b * T_val);
  double hc = pow(10.0, d_c + d_d * T_val);

  double rad= sqrt(2.0);
  double w= d_pc;
  double pr= rr;

  for (int k = 0; k < MOORE_NEIGHBORS; k++)
  {
    int ni = i + d_Xi[k];
    int nj = j + d_Xj[k];

    bool is_valid = (ni >= 0 && ni < rows && nj >= 0 && nj < cols);

    if (is_valid) {
      int idx_k = ni * cols + nj;
      double sz_k = sz[idx_k];
      h[k] = sh[idx_k];

      if (k < VON_NEUMANN_NEIGHBORS)
        z[k] = sz_k;
      else
        z[k] = sz0 - (sz0 - sz_k) / rad; 
    } 
  }

  H[0] = z[0];
  theta[0] = 0.0;
  eliminated[0] = false;

  for (int k = 1; k < MOORE_NEIGHBORS; k++)
  {
    eliminated[k] = true;
    H[k] = 0.0; 
    theta[k]=0.0;
    
    if (z[0] + h[0] > z[k] + h[k])
    {
      H[k] = z[k] + h[k];
      theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w);
      eliminated[k] = false;
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

    mf[mf_idx]=0.0;

    if (!eliminated[k] && h[0] > hc * cos(theta[k]))
    {
      mf[mf_idx] = pr * (avg - H[k]);
    }
  }
}


__global__ void massBalance_Global(double *sh, double *sh_next, double *st, double *st_next, double *mf)        
{


  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= rows || j >= cols) return;

  int idx = i * cols + j;

  const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};

  double initial_h = sh[idx];
  double initial_t = st[idx];

  double h_next = initial_h;
  double t_next = initial_h * initial_t;

  int layer_size = rows * cols;

  for (int n = 1; n < MOORE_NEIGHBORS; n++)
  {
    int ni = i + d_Xi[n];
    int nj = j + d_Xj[n];

    if (ni < 0 || ni >= rows || nj < 0 || nj >= cols)
    {
      continue;
    }

    int n_idx = ni * cols + nj;

    int out_layer = n - 1;
    double outFlow = mf[out_layer * layer_size + idx];

    int in_layer = inflowsIndices[n - 1];
    double inFlow = mf[in_layer * layer_size + n_idx];

    double neigh_t = st[n_idx];

    h_next += (inFlow - outFlow);

    t_next += (inFlow * neigh_t - outFlow * initial_t);
  }

  if (h_next > 0) {
    t_next /= h_next; 

    st_next[idx] = t_next;
    sh_next[idx] = h_next;
  }
}


__global__ void computeNewTemperatureAndSolidification_Global(
        double* __restrict__ sh, double*  __restrict__ sh_next, double* __restrict__ st, double* __restrict__ st_next, double* __restrict__ sz, double* __restrict__ sz_next, double* __restrict__ mhs, const bool *__restrict__ mb         
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= rows || j >= cols) return;

    int idx = i * cols + j;

    double h = sh[idx];
    double T = st[idx];
    double z = sz[idx];
    
    double h_out = h;
    double T_out = T;
    double z_out = z;

    if (h > 0.0 && !mb[idx]) {
        double T3 = T * T * T;
        double aus = 1.0 + (d_temp_factor * T3) / (d_temp_divisor * h);
        double nT = T / pow(aus, 1.0 / 3.0);  // Formula originale
        if (nT > d_ptsol) {
            T_out = nT;
        } else {
            z_out = z + h;
            h_out = 0.0;
            T_out = d_ptsol;
            mhs[idx] += h;
        }
    }

    sh_next[idx] = h_out;
    st_next[idx] = T_out;
    sz_next[idx] = z_out;
}


__global__ void boundaryConditions_Global(Sciara *sciara)
{
    
    double *sh = sciara->substates->Sh;
    double *st = sciara->substates->ST;
    bool *mb = sciara->substates->Mb;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= rows || j >= cols) return;

    int idx = i * cols + j;

    if (mb[idx])
    {
        sh[idx] = 0.0;
        st[idx] = 0.0;
    }
}
