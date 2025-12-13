#include "../../src/vent.h"
#include "../../src/Sciara.h"
#include "../../implementations/tiled_with_halos/kernel_tiled_with_halo.cuh"
#include "../../constants.cuh"  

#ifndef HALO
#define HALO 1
#endif
__global__ void CfAMo_Kernel(
    const double * __restrict__ sh, 
    const double * __restrict__ st, 
    const double * __restrict__ sz, 
    double * __restrict__ sh_next, 
    double * __restrict__ st_next) 
{
    int sharedWidth = blockDim.x; 
    int sharedHeight = blockDim.y;
    int sharedSize = sharedWidth * sharedHeight;

    extern __shared__ double shared_mem[];
    
    double *sh_s = shared_mem;
    double *sz_s = shared_mem + sharedSize;
    double *st_s = shared_mem + 2 * sharedSize;
    double *sh_accum = shared_mem + 3 * sharedSize;              
    double *en_accum = shared_mem + 4 * sharedSize; 

    int tc = threadIdx.x;
    int tr = threadIdx.y;
    int tid_s = tr * sharedWidth + tc;

    int j = blockIdx.x * (blockDim.x - 2 * HALO) + tc - HALO;
    int i = blockIdx.y * (blockDim.y - 2 * HALO) + tr - HALO;
    int idx = i * cols + j;

    bool is_valid_sim_cell = (i >= 0 && i < rows && j >= 0 && j < cols);

    if (is_valid_sim_cell) {
        sh_s[tid_s] = sh[idx];
        sz_s[tid_s] = sz[idx];
        st_s[tid_s] = st[idx];
    } else {
        sh_s[tid_s] = 0.0; 
        sz_s[tid_s] = 0.0; 
        st_s[tid_s] = 0.0;
    }

    __syncthreads();

    sh_accum[tid_s] = 0.0;
    en_accum[tid_s] = 0.0;
    
    double calculated_flows[8];
    #pragma unroll
    for(int k = 0; k < 8; k++) calculated_flows[k] = 0.0;

    double h0 = 0.0;
    double t0 = 0.0;

    if (is_valid_sim_cell) {
        h0 = sh_s[tid_s]; 
        
        if (h0 > 0.0) {
            t0 = st_s[tid_s]; 
            double sz0 = sz_s[tid_s];
            
            bool eliminated[MOORE_NEIGHBORS];
            double z[MOORE_NEIGHBORS];
            double h[MOORE_NEIGHBORS];
            double H[MOORE_NEIGHBORS];
            double theta[MOORE_NEIGHBORS];

            double rr = pow(10.0, d_a + d_b * t0);
            double hc = pow(10.0, d_c + d_d * t0);
            double rad = sqrt(2.0);

            double w = d_pc;
            double pr = rr;

            #pragma unroll
            for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                int ntc = tc + d_Xj[k];
                int ntr = tr + d_Xi[k];
                double sz_k, h_k;

                if (ntc >= 0 && ntc < sharedWidth && ntr >= 0 && ntr < sharedHeight) {
                    int nidx_s = ntr * sharedWidth + ntc;
                    h_k = sh_s[nidx_s];
                    sz_k = sz_s[nidx_s];
                } else {
                    int ni = i + d_Xi[k];
                    int nj = j + d_Xj[k];
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                        int nidx = ni * cols + nj;
                        h_k = sh[nidx];
                        sz_k = sz[nidx];
                    } else {
                        h_k = 0.0;
                        sz_k = sz0; 
                    }
                }
                
                h[k] = h_k;
                if (k < VON_NEUMANN_NEIGHBORS) z[k] = sz_k;
                else z[k] = sz0 - (sz0 - sz_k) / rad;
            }

            H[0] = z[0]; theta[0] = 0.0; eliminated[0] = false;
            
            #pragma unroll
            for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                if (z[0] + h[0] > z[k] + h[k]) {
                    H[k] = z[k] + h[k];
                    theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w);
                    eliminated[k] = false;
                } else {
                    eliminated[k] = true; H[k] = 0.0; theta[k] = 0.0;
                }
            }

            bool loop;
            double avg;
            int counter;
            do {
                loop = false;
                avg = h[0];
                counter = 0;
                for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                    if (!eliminated[k]) { avg += H[k]; counter++; }
                }
                if (counter != 0) avg /= (double)counter;
                for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                    if (!eliminated[k] && avg <= H[k]) {
                        eliminated[k] = true; loop = true;
                    }
                }
            } while (loop);
            
            #pragma unroll
            for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                if (!eliminated[k] && h[0] > hc * cos(theta[k])) {
                    calculated_flows[k-1] = pr * (avg - H[k]);
                }
            }
        }
    }

    __syncthreads();

    double total_outflow = 0.0;

    for (int step = 1; step < MOORE_NEIGHBORS; step++) { 
        
        double my_flow = calculated_flows[step - 1]; 
        total_outflow += my_flow;

        int tn_c = tc + d_Xj[step];
        int tn_r = tr + d_Xi[step];

        if (tn_c >= 0 && tn_c < sharedWidth && tn_r >= 0 && tn_r < sharedHeight) {
            int tid_s_neigh = tn_r * sharedWidth + tn_c;
            if (my_flow > 0.0) {
                double current_mass = sh_accum[tid_s_neigh];
                sh_accum[tid_s_neigh] = current_mass + my_flow;

                double current_energy = en_accum[tid_s_neigh];
                en_accum[tid_s_neigh] = current_energy + (my_flow * t0);
            }
        }
        
        __syncthreads();
    }

    bool internal_thread = (tc >= HALO && tc < blockDim.x - HALO && 
                            tr >= HALO && tr < blockDim.y - HALO);

    if (internal_thread && is_valid_sim_cell) {
        
        double inflow_mass = sh_accum[tid_s];
        double inflow_energy = en_accum[tid_s];
        
        double h_new = h0 + inflow_mass - total_outflow;
        double t_new = t0;

        if (h_new > 0.0) {
            double e_residual = (h0 - total_outflow) * t0;
            t_new = (e_residual + inflow_energy) / h_new;
            sh_next[idx] = h_new;
            st_next[idx] = t_new;
        }
    }
}