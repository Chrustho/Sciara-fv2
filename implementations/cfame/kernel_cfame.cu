#include "../../src/Sciara.h"
#include "kernel_cfame.cuh"

#define HALO 1
// il motivo dell'indice inverso è perché se un thread cede calore
// al thread a nord, allora quest'ultimo diciamo che riceve dal thread sud
__constant__ int _indice_inverso_k[] = {0, 4, 3, 2, 1, 7, 8, 5, 6};
__constant__ int _Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1}; // Xj: Moore neighborhood row coordinates (see below)
__constant__ int _Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1}; // Xj: Moore neighborhood col coordinates (see below)


__global__ void  computeOutflows_cfame(Sciara *sciara){

    int rows = sciara->domain->rows;
    int cols = sciara->domain->cols;

    double *sh = sciara->substates->Sh;
    double *h_next = sciara->substates->Sh_next;
    double *st = sciara->substates->ST;
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
    double *sh_s = shared_mem;
    double *st_s = shared_mem + sharedSize;
    double *sz_s = shared_mem + sharedSize * 2;
    double *f_s = shared_mem + sharedSize * 3;

    int tc = threadIdx.x;
    int tr = threadIdx.y;

    int j = blockIdx.x * blockDim.x + tc;
    int i = blockIdx.y * blockDim.y + tr;
    int idx = i * cols + j;

    int ts_c = tc + HALO; 
    int ts_r = tr + HALO;  
    int tid_s = ts_r * sharedWidth + ts_c;

    for(int k=0; k<8; k++) {
        f_s[(tid_s * 8) + k] = 0.0;
    }

    // CARICAMENTO TILE CENTRALE 
    if (i < rows && j < cols) {
        sh_s[tid_s] = sh[idx];
        st_s[tid_s] = st[idx];
        sz_s[tid_s] = sz[idx];
    }

    //  CARICAMENTO HALO 
    
    // Halo sinistro
    if (tc == 0) {
        int gi = i;
        int gj = j - HALO;
        int sid = ts_r * sharedWidth + (ts_c - HALO);
        
        for(int k=0; k<8; k++) f_s[(sid * 8) + k] = 0.0;
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
        
        for(int k=0; k<8; k++) f_s[(sid * 8) + k] = 0.0;
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
        
        for(int k=0; k<8; k++) f_s[(sid * 8) + k] = 0.0;
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
        for(int k=0; k<8; k++) f_s[(sid * 8) + k] = 0.0;
        
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
        for(int k=0; k<8; k++) f_s[(sid * 8) + k] = 0.0;
        
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
        for(int k=0; k<8; k++) f_s[(sid * 8) + k] = 0.0;
        
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
        for(int k=0; k<8; k++) f_s[(sid * 8) + k] = 0.0;
        
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
        for(int k=0; k<8; k++) f_s[(sid * 8) + k] = 0.0;
        
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
    double Pr[MOORE_NEIGHBORS];
    double w[MOORE_NEIGHBORS];

    double sz0 = sz_s[tid_s];
    double T_val = st_s[tid_s];

    double rr = pow(10.0, _a + _b * T_val);
    double hc = pow(10.0, _c + _d * T_val);

    for (int k = 0; k < MOORE_NEIGHBORS; k++)
    {
        int ni = i + _Xi[k];
        int nj = j + _Xj[k];

        bool is_valid = (ni >= 0 && ni < rows && nj >= 0 && nj < cols);

        if (is_valid) {
            int ts_r_k = ts_r + _Xi[k];
            int ts_c_k = ts_c + _Xj[k];
            int tid_s_k = ts_r_k * sharedWidth + ts_c_k;

            double sz_k = sz_s[tid_s_k];
            h[k] = sh_s[tid_s_k];

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
            theta[k] = 0.0;
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

    // parte di cfame, praticamente aggiungo tutto in shmem anziché global mf

    for (int k = 1; k < MOORE_NEIGHBORS; k++)
    {

      double flow = 0.0;
      if(!eliminated[k] && h[0] > hc*cos(theta[k])) {
        flow = Pr[k] * (avg - H[k]);
      }

      f_s[tid_s * 8 + k - 1] = flow;
    }


    __syncthreads();

    /* Ora posso usare i flussi in shmem calcolati per aggiornare tutti in massa */

    if( i < rows && j < cols) {
      double h_new = sh_s[tid_s];
      for (int k = 1; k < MOORE_NEIGHBORS; k++) {
        // caccia i flussi uscenti
        double out_flow = f_s[tid_s * 8 + k - 1];
        h_new -=out_flow;

        // aggiungi i flussi
        int ts_r_vicino = ts_r + _Xi[k];
        int ts_c_vicino = ts_c + _Xj[k];
        int tid_s_vicino = ts_r_vicino * sharedWidth + ts_c_vicino;


        int indice_inverso = _indice_inverso_k[k];
        double in_flow = f_s[(tid_s_vicino * 8) + (indice_inverso - 1)];
        h_new += in_flow;
      }
      h_next[idx] = h_new;
    }
}

