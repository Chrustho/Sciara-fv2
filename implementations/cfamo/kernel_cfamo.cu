#include "../../src/vent.h"
#include "../../src/Sciara.h"
#include "../../implementations/tiled_with_halos/kernel_tiled_with_halo.cuh"

// Assicurati che HALO sia definito
#ifndef HALO
#define HALO 1
#endif

// Moore Neighborhood relative coordinates (Centre + 8 neighbors)
__constant__ int _Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1}; 
__constant__ int _Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1}; 

__global__ void CfAMo_Kernel(Sciara *sciara) {

    int rows = sciara->domain->rows;
    int cols = sciara->domain->cols;

    double *sh = sciara->substates->Sh;
    double *st = sciara->substates->ST;
    double *sz = sciara->substates->Sz;
    double *mf = sciara->substates->Mf; 
    
    // Output
    double *sh_next = sciara->substates->Sh_next;
    double *st_next = sciara->substates->ST_next;

    double _a = sciara->parameters->a;
    double _b = sciara->parameters->b;
    double _c = sciara->parameters->c;
    double _d = sciara->parameters->d;
    double pc = sciara->parameters->Pc;

    int sharedWidth = blockDim.x; 
    int sharedHeight = blockDim.y;
    int sharedSize = sharedWidth * sharedHeight;

    extern __shared__ double shared_mem[];
    double *sh_accum = shared_mem;              
    double *en_accum = shared_mem + sharedSize; 
    int tc = threadIdx.x;
    int tr = threadIdx.y;
    int tid_s = tr * sharedWidth + tc;


    int j = blockIdx.x * (blockDim.x - 2 * HALO) + tc - HALO;
    int i = blockIdx.y * (blockDim.y - 2 * HALO) + tr - HALO;
    int idx = i * cols + j;

    sh_accum[tid_s] = 0.0;
    en_accum[tid_s] = 0.0;
    
    __syncthreads();

    bool is_valid_sim_cell = (i >= 0 && i < rows && j >= 0 && j < cols);
    
    double calculated_flows[8];
    for(int k=0; k<8; k++) calculated_flows[k] = 0.0;
    
    double h0 = 0.0;
    double t0 = 0.0;

    if (is_valid_sim_cell) {
        h0 = sh[idx]; 
        if (h0 > 0.0) {
            t0 = st[idx];
            double sz0 = sz[idx];
            
            bool eliminated[MOORE_NEIGHBORS];
            double z[MOORE_NEIGHBORS];
            double h[MOORE_NEIGHBORS];
            double H[MOORE_NEIGHBORS];
            double theta[MOORE_NEIGHBORS];
            double Pr[MOORE_NEIGHBORS];
            double w[MOORE_NEIGHBORS];

            double rr = pow(10.0, _a + _b * t0);
            double hc = pow(10.0, _c + _d * t0);

            // Caricamento Vicini da GLOBAL MEMORY
            for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                int ni = i + _Xi[k];
                int nj = j + _Xj[k];
                
                if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                    int nidx = ni * cols + nj;
                    h[k] = sh[nidx];
                    double sz_k = sz[nidx];
                    
                    if (k < VON_NEUMANN_NEIGHBORS) z[k] = sz_k;
                    else z[k] = sz0 - (sz0 - sz_k) / sqrt(2.0);
                }
                w[k] = pc;
                Pr[k] = rr;
            }

            H[0] = z[0]; theta[0] = 0.0; eliminated[0] = false;

            for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                if (z[0] + h[0] > z[k] + h[k]) {
                    H[k] = z[k] + h[k];
                    theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
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

            // Salvataggio locale flussi
            for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                if (!eliminated[k] && h[0] > hc * cos(theta[k])) {
                    calculated_flows[k-1] = Pr[k] * (avg - H[k]);
                }
            }
        }
    }


    
    double total_outflow = 0.0;

    for (int step = 1; step < MOORE_NEIGHBORS; step++) { 
        
        double my_flow = calculated_flows[step - 1]; 
        total_outflow += my_flow;

        int tn_c = tc + _Xj[step];
        int tn_r = tr + _Xi[step];

        if (tn_c >= 0 && tn_c < blockDim.x && tn_r >= 0 && tn_r < blockDim.y) {
            
            int tid_s_neigh = tn_r * sharedWidth + tn_c;

            if (my_flow > 0.0) {
                // Accumulo Massa
                double current_mass = sh_accum[tid_s_neigh];
                sh_accum[tid_s_neigh] = current_mass + my_flow;

                // Accumulo Energia (Massa * Temp)
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
        
        // Bilancio
        double h_new = h0 + inflow_mass - total_outflow;
        double t_new = t0;

        if (h_new > 0) {
             double e_residual = (h0 - total_outflow) * t0;
             t_new = (e_residual + inflow_energy) / h_new;
        }

        sh_next[idx] = h_new;
        st_next[idx] = t_new;
        
    }
}