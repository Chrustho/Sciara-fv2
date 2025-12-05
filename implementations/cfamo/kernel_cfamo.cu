#pragma once

#include "../../src/vent.h"
#include "../../src/Sciara.h"
#include "kernel_cfamo.cuh"

#define HALO 1

__constant__ int d_Xi_cfamo[] = {0, -1,  0,  0,  1, -1,  1,  1, -1};
__constant__ int d_Xj_cfamo[] = {0,  0, -1,  1,  0, -1, -1,  1,  1};


__global__ void CfAMo_Kernel(Sciara *sciara) {
    
    int rows = sciara->domain->rows;
    int cols = sciara->domain->cols;

    double *sh = sciara->substates->Sh;
    double *sh_next = sciara->substates->Sh_next;
    double *st = sciara->substates->ST;
    double *st_next = sciara->substates->ST_next;
    double *sz = sciara->substates->Sz;

    double _a = sciara->parameters->a;
    double _b = sciara->parameters->b;
    double _c = sciara->parameters->c;
    double _d = sciara->parameters->d;
    double pc = sciara->parameters->Pc;

    int sharedWidth = blockDim.x + 2 * HALO;
    int sharedHeight = blockDim.y + 2 * HALO;
    int sharedSize = sharedWidth * sharedHeight;

    extern __shared__ double shared_mem[];
    double *h_next_shared = shared_mem;
    double *t_next_shared = shared_mem + sharedSize;

    int tc = threadIdx.x;
    int tr = threadIdx.y;

    int j = blockIdx.x * blockDim.x + tc;
    int i = blockIdx.y * blockDim.y + tr;
    int idx = i * cols + j;

    int ts_c = tc + HALO;
    int ts_r = tr + HALO;
    int tid_s = ts_r * sharedWidth + ts_c;

    if (i < rows && j < cols) {
        h_next_shared[tid_s] = sh[idx];
        t_next_shared[tid_s] = sh[idx] * st[idx];  
    }

    //  halo sinistro
    if (tc == 0) {
        int gi = i;
        int gj = j - HALO;
        int sid = ts_r * sharedWidth + (ts_c - HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        }
        else {
            h_next_shared[sid] = 0.0;
            t_next_shared[sid] = 0.0;
        }
    }

    //  halo destro
    if (tc == blockDim.x - 1) {
        int gi = i;
        int gj = j + HALO;
        int sid = ts_r * sharedWidth + (ts_c + HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        }
    }

    //  halo superiore
    if (tr == 0) {
        int gi = i - HALO;
        int gj = j;
        int sid = (ts_r - HALO) * sharedWidth + ts_c;
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        }
        else {
            h_next_shared[sid] = 0.0;
            t_next_shared[sid] = 0.0;
        }
    }

    //  halo inferiore
    if (tr == blockDim.y - 1) {
        int gi = i + HALO;
        int gj = j;
        int sid = (ts_r + HALO) * sharedWidth + ts_c;
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        }
        else {
            h_next_shared[sid] = 0.0;
            t_next_shared[sid] = 0.0;
        }
    }

    //  angolo top-left
    if (tc == 0 && tr == 0) {
        int gi = i - HALO;
        int gj = j - HALO;
        int sid = (ts_r - HALO) * sharedWidth + (ts_c - HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        }
        else {
            h_next_shared[sid] = 0.0;
            t_next_shared[sid] = 0.0;
        }
    }

    //  angolo top-right
    if (tc == blockDim.x - 1 && tr == 0) {
        int gi = i - HALO;
        int gj = j + HALO;
        int sid = (ts_r - HALO) * sharedWidth + (ts_c + HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        }else {
            h_next_shared[sid] = 0.0;
            t_next_shared[sid] = 0.0;
        }
    }

    //  angolo bottom-left
    if (tc == 0 && tr == blockDim.y - 1) {
        int gi = i + HALO;
        int gj = j - HALO;
        int sid = (ts_r + HALO) * sharedWidth + (ts_c - HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        }
        else {
            h_next_shared[sid] = 0.0;
            t_next_shared[sid] = 0.0;
        }
    }

    //  angolo bottom-right
    if (tc == blockDim.x - 1 && tr == blockDim.y - 1) {
        int gi = i + HALO;
        int gj = j + HALO;
        int sid = (ts_r + HALO) * sharedWidth + (ts_c + HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        }
        else {
            h_next_shared[sid] = 0.0;
            t_next_shared[sid] = 0.0;
        }
    }

    __syncthreads();

    double myOutflows[NUMBER_OF_OUTFLOWS];
    for (int k = 0; k < NUMBER_OF_OUTFLOWS; k++) {
        myOutflows[k] = 0.0;
    }
    
    double myTemp = 0.0;

    if (i < rows && j < cols) {
        double h0 = sh[idx];
        myTemp = st[idx];

        if (h0 > 0.0) {
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

            for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                int ni = i + d_Xi_cfamo[k];
                int nj = j + d_Xj_cfamo[k];

                bool is_valid = (ni >= 0 && ni < rows && nj >= 0 && nj < cols);

                if (is_valid) {
                    int idx_k = ni * cols + nj;
                    double sz_k = sz[idx_k];
                    h[k] = sh[idx_k];

                    if (k < VON_NEUMANN_NEIGHBORS)
                        z[k] = sz_k;
                    else
                        z[k] = sz0 - (sz0 - sz_k) / sqrt(2.0);
                } else {
                    h[k] = 0.0;
                    z[k] = sz0;
                }

                w[k] = pc;
                Pr[k] = rr;
            }

            H[0] = z[0];
            theta[0] = 0.0;
            eliminated[0] = false;

            for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                if (z[0] + h[0] > z[k] + h[k]) {
                    H[k] = z[k] + h[k];
                    theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
                    eliminated[k] = false;
                } else {
                    eliminated[k] = true;
                    H[k] = 0.0;
                    theta[k] = 0.0;
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
                    if (!eliminated[k]) {
                        avg += H[k];
                        counter++;
                    }
                }

                if (counter != 0)
                    avg = avg / (double)counter;

                for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                    if (!eliminated[k] && avg <= H[k]) {
                        eliminated[k] = true;
                        loop = true;
                    }
                }
            } while (loop);

            for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                int outflow_idx = k - 1;

                if (!eliminated[k] && h[0] > hc * cos(theta[k])) {
                    myOutflows[outflow_idx] = Pr[k] * (avg - H[k]);
                }
            }
        }
    }

    
    if (i < rows && j < cols) {
        double flow = myOutflows[0];  
        
        if (flow > 0.0) {
            int ni = i + d_Xi_cfamo[1];  
            int nj = j + d_Xj_cfamo[1]; 
            
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                int ts_r_n = ts_r + d_Xi_cfamo[1];
                int ts_c_n = ts_c + d_Xj_cfamo[1];
                int tid_s_n = ts_r_n * sharedWidth + ts_c_n;
                
                h_next_shared[tid_s_n] += flow;
                t_next_shared[tid_s_n] += flow * myTemp;
                
                h_next_shared[tid_s] -= flow;
                t_next_shared[tid_s] -= flow * myTemp;
            }
        }
    }
    __syncthreads();

    if (i < rows && j < cols) {
        double flow = myOutflows[1];
        
        if (flow > 0.0) {
            int ni = i + d_Xi_cfamo[2];  // i
            int nj = j + d_Xj_cfamo[2];  // j - 1
            
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                int ts_r_n = ts_r + d_Xi_cfamo[2];
                int ts_c_n = ts_c + d_Xj_cfamo[2];
                int tid_s_n = ts_r_n * sharedWidth + ts_c_n;
                
                h_next_shared[tid_s_n] += flow;
                t_next_shared[tid_s_n] += flow * myTemp;
                h_next_shared[tid_s] -= flow;
                t_next_shared[tid_s] -= flow * myTemp;
            }
        }
    }
    __syncthreads();

    if (i < rows && j < cols) {
        double flow = myOutflows[2];
        
        if (flow > 0.0) {
            int ni = i + d_Xi_cfamo[3];  // i
            int nj = j + d_Xj_cfamo[3];  // j + 1
            
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                int ts_r_n = ts_r + d_Xi_cfamo[3];
                int ts_c_n = ts_c + d_Xj_cfamo[3];
                int tid_s_n = ts_r_n * sharedWidth + ts_c_n;
                
                h_next_shared[tid_s_n] += flow;
                t_next_shared[tid_s_n] += flow * myTemp;
                h_next_shared[tid_s] -= flow;
                t_next_shared[tid_s] -= flow * myTemp;
            }
        }
    }
    __syncthreads();

    if (i < rows && j < cols) {
        double flow = myOutflows[3];
        
        if (flow > 0.0) {
            int ni = i + d_Xi_cfamo[4];  // i + 1
            int nj = j + d_Xj_cfamo[4];  // j
            
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                int ts_r_n = ts_r + d_Xi_cfamo[4];
                int ts_c_n = ts_c + d_Xj_cfamo[4];
                int tid_s_n = ts_r_n * sharedWidth + ts_c_n;
                
                h_next_shared[tid_s_n] += flow;
                t_next_shared[tid_s_n] += flow * myTemp;
                h_next_shared[tid_s] -= flow;
                t_next_shared[tid_s] -= flow * myTemp;
            }
        }
    }
    __syncthreads();

    if (i < rows && j < cols) {
        double flow = myOutflows[4];
        
        if (flow > 0.0) {
            int ni = i + d_Xi_cfamo[5];  // i - 1
            int nj = j + d_Xj_cfamo[5];  // j - 1
            
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                int ts_r_n = ts_r + d_Xi_cfamo[5];
                int ts_c_n = ts_c + d_Xj_cfamo[5];
                int tid_s_n = ts_r_n * sharedWidth + ts_c_n;
                
                h_next_shared[tid_s_n] += flow;
                t_next_shared[tid_s_n] += flow * myTemp;
                h_next_shared[tid_s] -= flow;
                t_next_shared[tid_s] -= flow * myTemp;
            }
        }
    }
    __syncthreads();

    if (i < rows && j < cols) {
        double flow = myOutflows[5];
        
        if (flow > 0.0) {
            int ni = i + d_Xi_cfamo[6];  // i + 1
            int nj = j + d_Xj_cfamo[6];  // j - 1
            
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                int ts_r_n = ts_r + d_Xi_cfamo[6];
                int ts_c_n = ts_c + d_Xj_cfamo[6];
                int tid_s_n = ts_r_n * sharedWidth + ts_c_n;
                
                h_next_shared[tid_s_n] += flow;
                t_next_shared[tid_s_n] += flow * myTemp;
                h_next_shared[tid_s] -= flow;
                t_next_shared[tid_s] -= flow * myTemp;
            }
        }
    }
    __syncthreads();

    if (i < rows && j < cols) {
        double flow = myOutflows[6];
        
        if (flow > 0.0) {
            int ni = i + d_Xi_cfamo[7];  // i + 1
            int nj = j + d_Xj_cfamo[7];  // j + 1
            
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                int ts_r_n = ts_r + d_Xi_cfamo[7];
                int ts_c_n = ts_c + d_Xj_cfamo[7];
                int tid_s_n = ts_r_n * sharedWidth + ts_c_n;
                
                h_next_shared[tid_s_n] += flow;
                t_next_shared[tid_s_n] += flow * myTemp;
                h_next_shared[tid_s] -= flow;
                t_next_shared[tid_s] -= flow * myTemp;
            }
        }
    }
    __syncthreads();

    if (i < rows && j < cols) {
        double flow = myOutflows[7];
        
        if (flow > 0.0) {
            int ni = i + d_Xi_cfamo[8];  // i - 1
            int nj = j + d_Xj_cfamo[8];  // j + 1
            
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                int ts_r_n = ts_r + d_Xi_cfamo[8];
                int ts_c_n = ts_c + d_Xj_cfamo[8];
                int tid_s_n = ts_r_n * sharedWidth + ts_c_n;
                
                h_next_shared[tid_s_n] += flow;
                t_next_shared[tid_s_n] += flow * myTemp;
                h_next_shared[tid_s] -= flow;
                t_next_shared[tid_s] -= flow * myTemp;
            }
        }
    }
    __syncthreads();

    if (i < rows && j < cols) {
        double h_final = h_next_shared[tid_s];
        double t_final = t_next_shared[tid_s];
        
        if (h_final > 0.0) {
            sh_next[idx] = h_final;
            st_next[idx] = t_final / h_final;  
        }
    }
}