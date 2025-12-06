#pragma once

#include "../../src/vent.h"
#include "../../src/Sciara.h"
#include "kernel_cfamo.cuh"

#define HALO 1

__constant__ int d_Xi_cfamo[] = {0, -1,  0,  0,  1, -1,  1,  1, -1};
__constant__ int d_Xj_cfamo[] = {0,  0, -1,  1,  0, -1, -1,  1,  1};

// Funzione helper per calcolare outflows di una cella specifica
__device__ void computeCellOutflows(
    int cell_i, int cell_j,
    int rows, int cols,
    double *sz, double *sh, double *st,
    double _a, double _b, double _c, double _d, double pc,
    double outflows[NUMBER_OF_OUTFLOWS],
    double &cellTemp)
{
    for (int k = 0; k < NUMBER_OF_OUTFLOWS; k++) {
        outflows[k] = 0.0;
    }
    cellTemp = 0.0;

    if (cell_i < 0 || cell_i >= rows || cell_j < 0 || cell_j >= cols) return;

    int cell_idx = cell_i * cols + cell_j;
    double h0 = sh[cell_idx];
    
    if (h0 <= 0.0) return;

    cellTemp = st[cell_idx];
    double sz0 = sz[cell_idx];

    double rr = pow(10.0, _a + _b * cellTemp);
    double hc = pow(10.0, _c + _d * cellTemp);

    bool eliminated[MOORE_NEIGHBORS];
    double z[MOORE_NEIGHBORS];
    double h[MOORE_NEIGHBORS];
    double H[MOORE_NEIGHBORS];
    double theta[MOORE_NEIGHBORS];
    double w[MOORE_NEIGHBORS];
    double Pr[MOORE_NEIGHBORS];

    for (int k = 0; k < MOORE_NEIGHBORS; k++) {
        int ni = cell_i + d_Xi_cfamo[k];
        int nj = cell_j + d_Xj_cfamo[k];

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
            outflows[outflow_idx] = Pr[k] * (avg - H[k]);
        }
    }
}

// Funzione helper per accumulare outflows di una cella su shared memory
__device__ void accumulateOutflowsForDirection(
    int dir_k,  // direzione (1-8)
    int cell_i, int cell_j,
    int cell_ts_r, int cell_ts_c,
    int rows, int cols,
    int sharedWidth, int sharedHeight,
    double flow, double cellTemp,
    double *h_next_shared, double *t_next_shared)
{
    if (flow <= 0.0) return;

    int ni = cell_i + d_Xi_cfamo[dir_k];
    int nj = cell_j + d_Xj_cfamo[dir_k];

    if (ni < 0 || ni >= rows || nj < 0 || nj >= cols) return;

    int ts_r_n = cell_ts_r + d_Xi_cfamo[dir_k];
    int ts_c_n = cell_ts_c + d_Xj_cfamo[dir_k];

    // Verifica che il vicino sia in shared memory (tile + halo)
    if (ts_r_n >= 0 && ts_r_n < sharedHeight && ts_c_n >= 0 && ts_c_n < sharedWidth) {
        int tid_s_n = ts_r_n * sharedWidth + ts_c_n;
        h_next_shared[tid_s_n] += flow;
        t_next_shared[tid_s_n] += flow * cellTemp;
    }
}


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

    // =========================================================================
    // FASE 1: CARICAMENTO h e t*h in shared memory (tile + halo)
    // =========================================================================

    // Tile centrale
    if (i < rows && j < cols) {
        h_next_shared[tid_s] = sh[idx];
        t_next_shared[tid_s] = sh[idx] * st[idx];
    } else {
        h_next_shared[tid_s] = 0.0;
        t_next_shared[tid_s] = 0.0;
    }

    // Halo sinistro
    if (tc == 0) {
        int gi = i, gj = j - HALO;
        int sid = ts_r * sharedWidth + (ts_c - HALO);
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        } else {
            h_next_shared[sid] = 0.0;
            t_next_shared[sid] = 0.0;
        }
    }

    // Halo destro
    if (tc == blockDim.x - 1) {
        int gi = i, gj = j + HALO;
        int sid = ts_r * sharedWidth + (ts_c + HALO);
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        } else {
            h_next_shared[sid] = 0.0;
            t_next_shared[sid] = 0.0;
        }
    }

    // Halo superiore
    if (tr == 0) {
        int gi = i - HALO, gj = j;
        int sid = (ts_r - HALO) * sharedWidth + ts_c;
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        } else {
            h_next_shared[sid] = 0.0;
            t_next_shared[sid] = 0.0;
        }
    }

    // Halo inferiore
    if (tr == blockDim.y - 1) {
        int gi = i + HALO, gj = j;
        int sid = (ts_r + HALO) * sharedWidth + ts_c;
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        } else {
            h_next_shared[sid] = 0.0;
            t_next_shared[sid] = 0.0;
        }
    }

    // Angoli
    if (tc == 0 && tr == 0) {
        int gi = i - HALO, gj = j - HALO;
        int sid = (ts_r - HALO) * sharedWidth + (ts_c - HALO);
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        } else {
            h_next_shared[sid] = 0.0;
            t_next_shared[sid] = 0.0;
        }
    }
    if (tc == blockDim.x - 1 && tr == 0) {
        int gi = i - HALO, gj = j + HALO;
        int sid = (ts_r - HALO) * sharedWidth + (ts_c + HALO);
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        } else {
            h_next_shared[sid] = 0.0;
            t_next_shared[sid] = 0.0;
        }
    }
    if (tc == 0 && tr == blockDim.y - 1) {
        int gi = i + HALO, gj = j - HALO;
        int sid = (ts_r + HALO) * sharedWidth + (ts_c - HALO);
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        } else {
            h_next_shared[sid] = 0.0;
            t_next_shared[sid] = 0.0;
        }
    }
    if (tc == blockDim.x - 1 && tr == blockDim.y - 1) {
        int gi = i + HALO, gj = j + HALO;
        int sid = (ts_r + HALO) * sharedWidth + (ts_c + HALO);
        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            h_next_shared[sid] = sh[gidx];
            t_next_shared[sid] = sh[gidx] * st[gidx];
        } else {
            h_next_shared[sid] = 0.0;
            t_next_shared[sid] = 0.0;
        }
    }

    __syncthreads();

    // =========================================================================
    // FASE 2: CALCOLO OUTFLOWS PER TILE + HALO
    // =========================================================================

    // Outflows per la propria cella (tile)
    double myOutflows[NUMBER_OF_OUTFLOWS];
    double myTemp = 0.0;
    computeCellOutflows(i, j, rows, cols, sz, sh, st, _a, _b, _c, _d, pc, myOutflows, myTemp);

    // Outflows per le celle dell'halo (thread del bordo fanno extra work)
    double haloOutflows_left[NUMBER_OF_OUTFLOWS];
    double haloTemp_left = 0.0;
    double haloOutflows_right[NUMBER_OF_OUTFLOWS];
    double haloTemp_right = 0.0;
    double haloOutflows_top[NUMBER_OF_OUTFLOWS];
    double haloTemp_top = 0.0;
    double haloOutflows_bottom[NUMBER_OF_OUTFLOWS];
    double haloTemp_bottom = 0.0;
    double haloOutflows_topleft[NUMBER_OF_OUTFLOWS];
    double haloTemp_topleft = 0.0;
    double haloOutflows_topright[NUMBER_OF_OUTFLOWS];
    double haloTemp_topright = 0.0;
    double haloOutflows_bottomleft[NUMBER_OF_OUTFLOWS];
    double haloTemp_bottomleft = 0.0;
    double haloOutflows_bottomright[NUMBER_OF_OUTFLOWS];
    double haloTemp_bottomright = 0.0;

    // Inizializza tutti a zero
    for (int k = 0; k < NUMBER_OF_OUTFLOWS; k++) {
        haloOutflows_left[k] = 0.0;
        haloOutflows_right[k] = 0.0;
        haloOutflows_top[k] = 0.0;
        haloOutflows_bottom[k] = 0.0;
        haloOutflows_topleft[k] = 0.0;
        haloOutflows_topright[k] = 0.0;
        haloOutflows_bottomleft[k] = 0.0;
        haloOutflows_bottomright[k] = 0.0;
    }

    // Thread bordo sinistro calcola per halo sinistro
    if (tc == 0) {
        computeCellOutflows(i, j - 1, rows, cols, sz, sh, st, _a, _b, _c, _d, pc, haloOutflows_left, haloTemp_left);
    }
    // Thread bordo destro calcola per halo destro
    if (tc == blockDim.x - 1) {
        computeCellOutflows(i, j + 1, rows, cols, sz, sh, st, _a, _b, _c, _d, pc, haloOutflows_right, haloTemp_right);
    }
    // Thread bordo superiore calcola per halo superiore
    if (tr == 0) {
        computeCellOutflows(i - 1, j, rows, cols, sz, sh, st, _a, _b, _c, _d, pc, haloOutflows_top, haloTemp_top);
    }
    // Thread bordo inferiore calcola per halo inferiore
    if (tr == blockDim.y - 1) {
        computeCellOutflows(i + 1, j, rows, cols, sz, sh, st, _a, _b, _c, _d, pc, haloOutflows_bottom, haloTemp_bottom);
    }
    // Angoli
    if (tc == 0 && tr == 0) {
        computeCellOutflows(i - 1, j - 1, rows, cols, sz, sh, st, _a, _b, _c, _d, pc, haloOutflows_topleft, haloTemp_topleft);
    }
    if (tc == blockDim.x - 1 && tr == 0) {
        computeCellOutflows(i - 1, j + 1, rows, cols, sz, sh, st, _a, _b, _c, _d, pc, haloOutflows_topright, haloTemp_topright);
    }
    if (tc == 0 && tr == blockDim.y - 1) {
        computeCellOutflows(i + 1, j - 1, rows, cols, sz, sh, st, _a, _b, _c, _d, pc, haloOutflows_bottomleft, haloTemp_bottomleft);
    }
    if (tc == blockDim.x - 1 && tr == blockDim.y - 1) {
        computeCellOutflows(i + 1, j + 1, rows, cols, sz, sh, st, _a, _b, _c, _d, pc, haloOutflows_bottomright, haloTemp_bottomright);
    }

    // =========================================================================
    // FASE 3: ACCUMULO - 8 direzioni, una alla volta
    // =========================================================================

    // Per ogni direzione k (1-8):
    for (int dir_k = 1; dir_k < MOORE_NEIGHBORS; dir_k++) {
        int outflow_idx = dir_k - 1;

        // Accumula per la propria cella (tile)
        accumulateOutflowsForDirection(dir_k, i, j, ts_r, ts_c,
            rows, cols, sharedWidth, sharedHeight,
            myOutflows[outflow_idx], myTemp,
            h_next_shared, t_next_shared);

        // Accumula per le celle dell'halo
        if (tc == 0) {
            accumulateOutflowsForDirection(dir_k, i, j - 1, ts_r, ts_c - 1,
                rows, cols, sharedWidth, sharedHeight,
                haloOutflows_left[outflow_idx], haloTemp_left,
                h_next_shared, t_next_shared);
        }
        if (tc == blockDim.x - 1) {
            accumulateOutflowsForDirection(dir_k, i, j + 1, ts_r, ts_c + 1,
                rows, cols, sharedWidth, sharedHeight,
                haloOutflows_right[outflow_idx], haloTemp_right,
                h_next_shared, t_next_shared);
        }
        if (tr == 0) {
            accumulateOutflowsForDirection(dir_k, i - 1, j, ts_r - 1, ts_c,
                rows, cols, sharedWidth, sharedHeight,
                haloOutflows_top[outflow_idx], haloTemp_top,
                h_next_shared, t_next_shared);
        }
        if (tr == blockDim.y - 1) {
            accumulateOutflowsForDirection(dir_k, i + 1, j, ts_r + 1, ts_c,
                rows, cols, sharedWidth, sharedHeight,
                haloOutflows_bottom[outflow_idx], haloTemp_bottom,
                h_next_shared, t_next_shared);
        }
        if (tc == 0 && tr == 0) {
            accumulateOutflowsForDirection(dir_k, i - 1, j - 1, ts_r - 1, ts_c - 1,
                rows, cols, sharedWidth, sharedHeight,
                haloOutflows_topleft[outflow_idx], haloTemp_topleft,
                h_next_shared, t_next_shared);
        }
        if (tc == blockDim.x - 1 && tr == 0) {
            accumulateOutflowsForDirection(dir_k, i - 1, j + 1, ts_r - 1, ts_c + 1,
                rows, cols, sharedWidth, sharedHeight,
                haloOutflows_topright[outflow_idx], haloTemp_topright,
                h_next_shared, t_next_shared);
        }
        if (tc == 0 && tr == blockDim.y - 1) {
            accumulateOutflowsForDirection(dir_k, i + 1, j - 1, ts_r + 1, ts_c - 1,
                rows, cols, sharedWidth, sharedHeight,
                haloOutflows_bottomleft[outflow_idx], haloTemp_bottomleft,
                h_next_shared, t_next_shared);
        }
        if (tc == blockDim.x - 1 && tr == blockDim.y - 1) {
            accumulateOutflowsForDirection(dir_k, i + 1, j + 1, ts_r + 1, ts_c + 1,
                rows, cols, sharedWidth, sharedHeight,
                haloOutflows_bottomright[outflow_idx], haloTemp_bottomright,
                h_next_shared, t_next_shared);
        }

        __syncthreads();
    }

    // =========================================================================
    // FASE 4: SOTTRAI OUTFLOWS E SCRIVI RISULTATO (solo tile)
    // =========================================================================

    if (i < rows && j < cols) {
        double total_outflow_h = 0.0;
        double total_outflow_t = 0.0;
        for (int k = 0; k < NUMBER_OF_OUTFLOWS; k++) {
            total_outflow_h += myOutflows[k];
            total_outflow_t += myOutflows[k] * myTemp;
        }

        double h_final = h_next_shared[tid_s] - total_outflow_h;
        double t_final = t_next_shared[tid_s] - total_outflow_t;

        if (h_final > 0.0) {
            sh_next[idx] = h_final;
            st_next[idx] = t_final / h_final;
        } else {
            sh_next[idx] = 0.0;
            st_next[idx] = 0.0;
        }
    }
}