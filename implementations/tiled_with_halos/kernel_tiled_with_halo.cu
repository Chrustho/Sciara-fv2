#pragma once

#include "../../src/vent.h"
#include "../../src/Sciara.h"
#include "../../implementations/tiled_with_halos/kernel_tiled_with_halo.cuh"

#define HALO 1  


__constant__ int _Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1}; // Xj: Moore neighborhood row coordinates (see below)
__constant__ int _Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1}; // Xj: Moore neighborhood col coordinates (see below)


__global__ void computeOutflows_Tiled_wH(
    Sciara *sciara, const unsigned int tileX, const unsigned int tileY){

    int rows = sciara->domain->rows;
    int cols = sciara->domain->cols;

    double *sh = sciara->substates->Sh;
    double *st = sciara->substates->ST;
    double *sz = sciara->substates->Sz;
    double *mf = sciara->substates->Mf;

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

    int tc = threadIdx.x;
    int tr = threadIdx.y;

    int j = blockIdx.x * blockDim.x + tc;
    int i = blockIdx.y * blockDim.y + tr;
    int idx = i * cols + j;

    int ts_c = tc + HALO; 
    int ts_r = tr + HALO;  
    int tid_s = ts_r * sharedWidth + ts_c;

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





__global__ void massBalance_Tiled_wH(
    Sciara *sciara, const unsigned int tileX, const unsigned int tileY){

}


__global__ void computeNewTemperatureAndSolidification_Tiled_wH();

__global__ void boundaryConditions_Tiled_wH();