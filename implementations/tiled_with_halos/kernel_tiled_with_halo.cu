#include "../../src/vent.h"
#include "../../src/Sciara.h"
#include "../../implementations/tiled_with_halos/kernel_tiled_with_halo.cuh"
#include "../../constants.cuh"  


#define HALO 1  


__global__ void computeOutflows_Tiled_wH(
        double *sh, double *st, double *sz, double *mf){

    int sharedWidth = blockDim.x + 2 * HALO;   
    int sharedHeight = blockDim.y + 2 * HALO;  
    int sharedSize = sharedWidth * sharedHeight;  

    extern __shared__ double shared_mem[];
    double *sh_s = shared_mem;
    double *st_s = shared_mem + sharedSize;
    double *sz_s = shared_mem + sharedSize * 2;

    int tc = threadIdx.x;
    int tr = threadIdx.y;

    int j = blockIdx.x * blockDim.x + tc;
    int i = blockIdx.y * blockDim.y + tr;
    int idx = i * cols + j;

    int ts_c = tc + HALO; 
    int ts_r = tr + HALO;  
    int tid_s = ts_r * sharedWidth + ts_c;

    if (i < rows && j < cols) {
        sh_s[tid_s] = sh[idx];
        st_s[tid_s] = st[idx];
        sz_s[tid_s] = sz[idx];
    }

    
    // Halo sinistro
    if (tc == 0) {
        int gi = i;
        int gj = j - HALO;
        int sid = ts_r * sharedWidth + (ts_c - HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            sz_s[sid] = sz[gidx];
        }
    }

    // Halo destro 
    if (tc == blockDim.x - 1) {
        int gi = i;
        int gj = j + HALO;
        int sid = ts_r * sharedWidth + (ts_c + HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            sz_s[sid] = sz[gidx];
        }
    }

    // Halo superiore 
    if (tr == 0) {
        int gi = i - HALO;
        int gj = j;
        int sid = (ts_r - HALO) * sharedWidth + ts_c;
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            sz_s[sid] = sz[gidx];
        }
    }

    // Halo inferiore 
    if (tr == blockDim.y - 1) {
        int gi = i + HALO;
        int gj = j;
        int sid = (ts_r + HALO) * sharedWidth + ts_c;
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            sz_s[sid] = sz[gidx];
        }
    }

    // Halo angoli 

    // Angolo top-left
    if (tc == 0 && tr == 0) {
        int gi = i - HALO;
        int gj = j - HALO;
        int sid = (ts_r - HALO) * sharedWidth + (ts_c - HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            sz_s[sid] = sz[gidx];
        }
    }

    // Angolo top-right
    if (tc == blockDim.x - 1 && tr == 0) {
        int gi = i - HALO;
        int gj = j + HALO;
        int sid = (ts_r - HALO) * sharedWidth + (ts_c + HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            sz_s[sid] = sz[gidx];
        }
    }

    // Angolo bottom-left
    if (tc == 0 && tr == blockDim.y - 1) {
        int gi = i + HALO;
        int gj = j - HALO;
        int sid = (ts_r + HALO) * sharedWidth + (ts_c - HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            sz_s[sid] = sz[gidx];
        }
    }

    // Angolo bottom-right
    if (tc == blockDim.x - 1 && tr == blockDim.y - 1) {
        int gi = i + HALO;
        int gj = j + HALO;
        int sid = (ts_r + HALO) * sharedWidth + (ts_c + HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            sz_s[sid] = sz[gidx];
        }
    }

    __syncthreads();


    if (i >= rows || j >= cols) return;

    double h0 = sh_s[tid_s];
    if (h0 <= 0.0) return;

    bool eliminated[MOORE_NEIGHBORS];
    double z[MOORE_NEIGHBORS];
    double h[MOORE_NEIGHBORS];
    double H[MOORE_NEIGHBORS];
    double theta[MOORE_NEIGHBORS];

    double sz0 = sz_s[tid_s];
    double T_val = st_s[tid_s];

    double rr = pow(10.0, d_a + d_b * T_val);
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
            int ts_r_k = ts_r + d_Xi[k];
            int ts_c_k = ts_c + d_Xj[k];
            int tid_s_k = ts_r_k * sharedWidth + ts_c_k;

            double sz_k = sz_s[tid_s_k];
            h[k] = sh_s[tid_s_k];

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
        theta[k] = 0.0;

        if (z[0] + h[0] > z[k] + h[k])
        {
            H[k] = z[k] + h[k];
            theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) /w);
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

        mf[mf_idx] = 0.0;

        if (!eliminated[k] && h[0] > hc * cos(theta[k]))
        {
            mf[mf_idx] = pr * (avg - H[k]);
        }
    }
}


__global__ void massBalance_Tiled_wH(
    double *sh, double *sh_next, double *st, double *st_next, double *mf) {


    int sharedWidth = blockDim.x + 2 * HALO;   
    int sharedHeight = blockDim.y + 2 * HALO;  
    int sharedSize = sharedWidth * sharedHeight;  

    extern __shared__ double shared_mem[];
    double *sh_s = shared_mem;
    double *st_s = shared_mem + sharedSize;
    double *mf_s = shared_mem + sharedSize * 2;

    int tc = threadIdx.x;
    int tr = threadIdx.y;

    int j = blockIdx.x * blockDim.x + tc;
    int i = blockIdx.y * blockDim.y + tr;
    int idx = i * cols + j;

    int ts_c = tc + HALO;
    int ts_r = tr + HALO;
    int tid_s = ts_r * sharedWidth + ts_c;

    int layer_size = rows * cols;

    //  CARICAMENTO TILE CENTRALE 
    if (i < rows && j < cols) {
        int gidx = i * cols + j;
        sh_s[tid_s] = sh[gidx];
        st_s[tid_s] = st[gidx];
        for (int layer = 0; layer < NUMBER_OF_OUTFLOWS; layer++) {
            mf_s[layer * sharedSize + tid_s] = mf[layer * layer_size + gidx];
        }
    }

    //  CARICAMENTO HALO SINISTRO 
    if (tc == 0) {
        int gi = i;
        int gj = j - HALO;
        int sid = ts_r * sharedWidth + (ts_c - HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            for (int layer = 0; layer < NUMBER_OF_OUTFLOWS; layer++) {
                mf_s[layer * sharedSize + sid] = mf[layer * layer_size + gidx];
            }
        }
    }

    //  CARICAMENTO HALO DESTRO 
    if (tc == blockDim.x - 1) {
        int gi = i;
        int gj = j + HALO;
        int sid = ts_r * sharedWidth + (ts_c + HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            for (int layer = 0; layer < NUMBER_OF_OUTFLOWS; layer++) {
                mf_s[layer * sharedSize + sid] = mf[layer * layer_size + gidx];
            }
        }
    }

    //  CARICAMENTO HALO SUPERIORE 
    if (tr == 0) {
        int gi = i - HALO;
        int gj = j;
        int sid = (ts_r - HALO) * sharedWidth + ts_c;
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            for (int layer = 0; layer < NUMBER_OF_OUTFLOWS; layer++) {
                mf_s[layer * sharedSize + sid] = mf[layer * layer_size + gidx];
            }
        } 
    }

    //  CARICAMENTO HALO INFERIORE 
    if (tr == blockDim.y - 1) {
        int gi = i + HALO;
        int gj = j;
        int sid = (ts_r + HALO) * sharedWidth + ts_c;
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            for (int layer = 0; layer < NUMBER_OF_OUTFLOWS; layer++) {
                mf_s[layer * sharedSize + sid] = mf[layer * layer_size + gidx];
            }
        }
    }

    //  CARICAMENTO ANGOLO TOP-LEFT 
    if (tc == 0 && tr == 0) {
        int gi = i - HALO;
        int gj = j - HALO;
        int sid = (ts_r - HALO) * sharedWidth + (ts_c - HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            for (int layer = 0; layer < NUMBER_OF_OUTFLOWS; layer++) {
                mf_s[layer * sharedSize + sid] = mf[layer * layer_size + gidx];
            }
        }
    }

    //  CARICAMENTO ANGOLO TOP-RIGHT 
    if (tc == blockDim.x - 1 && tr == 0) {
        int gi = i - HALO;
        int gj = j + HALO;
        int sid = (ts_r - HALO) * sharedWidth + (ts_c + HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            for (int layer = 0; layer < NUMBER_OF_OUTFLOWS; layer++) {
                mf_s[layer * sharedSize + sid] = mf[layer * layer_size + gidx];
            }
        }
    }

    //  CARICAMENTO ANGOLO BOTTOM-LEFT 
    if (tc == 0 && tr == blockDim.y - 1) {
        int gi = i + HALO;
        int gj = j - HALO;
        int sid = (ts_r + HALO) * sharedWidth + (ts_c - HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            for (int layer = 0; layer < NUMBER_OF_OUTFLOWS; layer++) {
                mf_s[layer * sharedSize + sid] = mf[layer * layer_size + gidx];
            }
        }
    }

    //  CARICAMENTO ANGOLO BOTTOM-RIGHT 
    if (tc == blockDim.x - 1 && tr == blockDim.y - 1) {
        int gi = i + HALO;
        int gj = j + HALO;
        int sid = (ts_r + HALO) * sharedWidth + (ts_c + HALO);
        
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            sh_s[sid] = sh[gidx];
            st_s[sid] = st[gidx];
            for (int layer = 0; layer < NUMBER_OF_OUTFLOWS; layer++) {
                mf_s[layer * sharedSize + sid] = mf[layer * layer_size + gidx];
            }
        }
    }

    __syncthreads();

    //  COMPUTAZIONE 

    if (i >= rows || j >= cols) return;

    const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};

    double initial_h = sh_s[tid_s];
    double initial_t = st_s[tid_s];

    double h_next = initial_h;
    double t_next = initial_h * initial_t;

    for (int n = 1; n < MOORE_NEIGHBORS; n++)
    {
        int ni = i + d_Xi[n];
        int nj = j + d_Xj[n];

        if (ni < 0 || ni >= rows || nj < 0 || nj >= cols)
            continue;

        int ts_r_n = ts_r + d_Xi[n];
        int ts_c_n = ts_c + d_Xj[n];
        int tid_s_n = ts_r_n * sharedWidth + ts_c_n;

        int out_layer = n - 1;
        double outFlow = mf_s[out_layer * sharedSize + tid_s];

        int in_layer = inflowsIndices[n - 1];
        double inFlow = mf_s[in_layer * sharedSize + tid_s_n];

        double neigh_t = st_s[tid_s_n];

        h_next += (inFlow - outFlow);
        t_next += (inFlow * neigh_t - outFlow * initial_t);
    }

    if (h_next > 0) {
        t_next /= h_next;
        st_next[idx] = t_next;
        sh_next[idx] = h_next;
    }
}


__global__ void computeNewTemperatureAndSolidification_Tiled_wH();
// In questo caso il kernel non beneficia del tiling perchè accede solo alla propria cella 
//e non a quella dei vicini

__global__ void boundaryConditions_Tiled_wH();
// In questo caso il kernel non beneficia del tiling perchè accede solo alla propria cella 
//e non a quella dei vicini
