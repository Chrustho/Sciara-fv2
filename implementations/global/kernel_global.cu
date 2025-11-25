#include "../../src/vent.h"
#include "../../src/Sciara.h"

__global__ void emitLava_Global(
    int r, 
    int c, 
    TVent *vents,        
    int num_vents,        
    double elapsed_time, 
    double Pclock, 
    double emission_time, 
    double &total_emitted_lava, 
    double Pac, 
    double PTvent, 
    double *Sh, 
    double *Sh_next, 
    double *ST_next
)
{
    int i= blockIdx.y*blockDim.y+threadIdx.y;
    int j=blockIdx.x*blockDim.x+threadIdx.x;

    int idx= i*c+j;

    if (i < r || j < c)
    {
        double curr= Sh[idx];
        for (size_t k = 0; k < num_vents; k++)
        {
            if (i==vents[k].y() && j==vents[k].x())
            {
                double outThickenss = vents[k].thickness(elapsed_time, Pclock, emission_time, Pac);
                Sh_next[idx]= curr+outThickenss;
                ST_next[idx]=PTvent;

                // dovremmo implementare somma atomica
                total_emitted_lava+=outThickenss;
            }
            
        }
        
    }
    
}

__global__ void computeOutflows_Global(
    int r, 
    int c, 
    int *Xi, 
    int *Xj, 
    double *Sz, 
    double *Sh, 
    double *ST, 
    double *Mf, 
    double Pc, 
    double _a, 
    double _b, 
    double _c, 
    double _d
)
{
    bool eliminated[MOORE_NEIGHBORS];
    double z[MOORE_NEIGHBORS];
    double h[MOORE_NEIGHBORS];
    double H[MOORE_NEIGHBORS];
    double theta[MOORE_NEIGHBORS];
    double w[MOORE_NEIGHBORS];  // Distances between central and adjecent cells
    double Pr[MOORE_NEIGHBORS]; // Relaiation rate arraj
    double f[MOORE_NEIGHBORS];
    bool loop;
    int counter;
    double sz0, sz, T, avg, rr, hc;

    int i= blockIdx.y*blockDim.y+threadIdx.y;
    int j=blockIdx.x*blockDim.x+threadIdx.x;

    int idx= i*c+j;
    double T= ST[idx];
    double rr= __powf64( _a + _b * T, 10);
    double hc=__powf64( _c + _d * T,10);

    for (size_t k = 0; k < MOORE_NEIGHBORS; k++)
    {
        double sz0= Sz[idx];
        int idx_k= (i+Xi[k])*c+(j+Xj[k]);
        double sz= Sz[idx_k];
        h[k]= sz;
        w[k]=Pc;
        Pr[k]=rr;
        if (k < VON_NEUMANN_NEIGHBORS)
            z[k] = sz;
        else
            z[k] = sz0 - (sz0 - sz) / sqrtf32(2.0); 
    }

    H[0] = z[0];
    theta[0] = 0;
    eliminated[0] = false;

    for (size_t k = 1; k < MOORE_NEIGHBORS; k++)
    {
        if (z[0] + h[0] > z[k] + h[k])
        {
            H[k] = z[k] + h[k];
            theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
            eliminated[k] = false;
        }
        else
        {
            // H[k] = 0;
            // theta[k] = 0;
            eliminated[k] = true;
        }
    }
    do
  {
    loop = false;
    avg = h[0];
    counter = 0;
    for (int k = 0; k < MOORE_NEIGHBORS; k++)
      if (!eliminated[k])
      {
        avg += H[k];
        counter++;
      }
    if (counter != 0)
      avg = avg / double(counter);
    for (int k = 0; k < MOORE_NEIGHBORS; k++)
      if (!eliminated[k] && avg <= H[k])
      {
        eliminated[k] = true;
        loop = true;
      }
  } while (loop);

  for (int k = 1; k < MOORE_NEIGHBORS; k++)
    if (!eliminated[k] && h[0] > hc * cos(theta[k]))
      BUF_SET(Mf, r, c, k - 1, i, j, Pr[k] * (avg - H[k]));
    else
      BUF_SET(Mf, r, c, k - 1, i, j, 0.0);

}


__global__ void massBalance_Global(
    int r, 
    int c, 
    int *Xi, 
    int *Xj, 
    double *Sh, 
    double *Sh_next, 
    double *ST, 
    double *ST_next, 
    double *Mf
);

__global__ void computeNewTemperatureAndSolidification_Global(
    int r, 
    int c, 
    double Pepsilon, 
    double Psigma, 
    double Pclock, 
    double Pcool, 
    double Prho, 
    double Pcv, 
    double Pac, 
    double PTsol, 
    double *Sz, 
    double *Sz_next, 
    double *Sh, 
    double *Sh_next, 
    double *ST, 
    double *ST_next, 
    double *Mf, 
    double *Mhs, 
    bool *Mb
);

__global__ void boundaryConditions_Global(
    int r, 
    int c, 
    double *Mf, 
    bool *Mb, 
    double *Sh, 
    double *Sh_next, 
    double *ST, 
    double *ST_next
);

__global__ void reduceAdd_Global(
    int r, 
    int c, 
    double *buffer,      
    double *partial_sums
);

