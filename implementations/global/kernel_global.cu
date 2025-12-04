#include "../../src/vent.h"
#include "../../src/Sciara.h"
#include "kernel_global.cuh"

__global__ void computeOutflows_Global(
    Sciara *sciara)
{
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

  // Parametri
  double *mf=sciara->substates->Mf;
  double pc=sciara->parameters->Pc;

  // a, b, c, d
  double _a= sciara->parameters->a;
  double _b=sciara->parameters->b;
  double _c=sciara->parameters->c;
  double _d=sciara->parameters->d;



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
  double Pr[MOORE_NEIGHBORS]; 
  double w[MOORE_NEIGHBORS];

  double sz0 = sz[idx];
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


__global__ void massBalance_Global(Sciara *sciara)        
{
  // Parametri del dominio
  int rows = sciara->domain->rows;
  int cols = sciara->domain->cols;

  // Vicini
  int *xi = sciara->X->Xi;
  int *xj = sciara->X->Xj;

  // Buffers
  double *sh = sciara->substates->Sh;
  double *sh_next = sciara->substates->Sh_next;
  double *st = sciara->substates->ST;
  double *st_next = sciara->substates->ST_next;
  double *mf = sciara->substates->Mf;

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= rows || j >= cols) return;

  int idx = i * cols + j;

  const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};

  double initial_h = sh[idx];
  double initial_t = st[idx];

  double h_next = initial_h;
  double t_next = initial_h * initial_t; // Energia termica attuale

  int layer_size = rows * cols;

  for (int n = 1; n < MOORE_NEIGHBORS; n++)
  {
    int ni = i + xi[n];
    int nj = j + xj[n];

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
    Sciara *sciara         
)
{
    // Parametri del dominio
    int rows= sciara->domain->rows;
    int cols= sciara->domain->cols;

    // Parametri

    double pepsilon=sciara->parameters->Pepsilon;
    double psigma=sciara->parameters->Psigma;
    double pclock=sciara->parameters->Pclock;
    double pcool=sciara->parameters->Pcool;
    double prho=sciara->parameters->Prho;
    double pcv=sciara->parameters->Pcv;
    double pac=sciara->parameters->Pac;
    double ptsol=sciara->parameters->PTsol;

    // Buffers
    double *sh=sciara->substates->Sh;
    double *sh_next=sciara->substates->Sh_next;
    double *st= sciara->substates->ST;
    double *st_next=sciara->substates->ST_next;
    double *sz=sciara->substates->Sz;
    double *sz_next=sciara->substates->Sz_next;

    double *mf=sciara->substates->Mf;
    double *mhs=sciara->substates->Mhs;
    bool *mb=sciara->substates->Mb;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= rows|| j >= cols) return;

    int idx = i * cols + j;

    double h = sh[idx];
    bool is_border = mb[idx];
    double T = st[idx];
    double z = sz[idx];

    sciara->substates->Sz_next[idx] = z;
    sciara->substates->Sh_next[idx] = h;
    sciara->substates->ST_next[idx] = T;

    if (h > 0.0 && !is_border)
    {
        double numerator = 3.0 * pow(T, 3.0) * pepsilon * psigma * pclock * pcool;
        double denominator = prho * pcv * h * pac;
        
        double aus = 1.0 + (numerator / denominator);

        double nT = T / pow(aus, 1.0/3.0);

        if (nT > ptsol) 
        {
            st_next[idx] = nT;
        } 
        else 
        {   
            sz_next[idx] = z + h;   
            sh_next[idx] = 0.0;     
            st_next[idx] = ptsol;   
            
            mhs[idx] = mhs[idx] + h;
        }
    }
}





__global__ void boundaryConditions_Global(Sciara *sciara)
{
    int rows = sciara->domain->rows;
    int cols = sciara->domain->cols;
    
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
