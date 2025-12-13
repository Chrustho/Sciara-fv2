#pragma once

#include "../../src/vent.h"
#include "../../src/Sciara.h"
#include "kernel_cfame.cuh"
#include "../../constants.cuh"  


#define HALO 1


__global__ void CfAMe_Kernel(const double* __restrict__ sh,
    const double* __restrict__ st,
    const double* __restrict__ sz,
    double* __restrict__ sh_next,
    double* __restrict__ st_next) {

    int sharedWidth = blockDim.x + 2 * HALO;
    int sharedHeight = blockDim.y + 2 * HALO;
    int sharedSize = sharedWidth * sharedHeight;

    extern __shared__ double f_shared[];

    int tc = threadIdx.x;
    int tr = threadIdx.y;

    int j = blockIdx.x * blockDim.x + tc;
    int i = blockIdx.y * blockDim.y + tr;
    int idx = i * cols + j;

    int ts_c = tc + HALO;
    int ts_r = tr + HALO;
    int tid_s = ts_r * sharedWidth + ts_c;


    if (i < rows && j < cols) {
        for (int k = 0; k < NUMBER_OF_OUTFLOWS; k++) {
            f_shared[k * sharedSize + tid_s] = 0.0;
        }
    }

    if (tc == 0) {
        int sid = ts_r * sharedWidth + (ts_c - HALO);
        
        for (int k = 0; k < NUMBER_OF_OUTFLOWS; k++) {
            f_shared[k * sharedSize + sid] = 0.0;
        }
    }

    if (tc == blockDim.x - 1) {
        int sid = ts_r * sharedWidth + (ts_c + HALO);
        
        for (int k = 0; k < NUMBER_OF_OUTFLOWS; k++) {
            f_shared[k * sharedSize + sid] = 0.0;
        }
    }

    if (tr == 0) {
        int sid = (ts_r - HALO) * sharedWidth + ts_c;
        
        for (int k = 0; k < NUMBER_OF_OUTFLOWS; k++) {
            f_shared[k * sharedSize + sid] = 0.0;
        }
    }

    if (tr == blockDim.y - 1) {
        int sid = (ts_r + HALO) * sharedWidth + ts_c;
        
        for (int k = 0; k < NUMBER_OF_OUTFLOWS; k++) {
            f_shared[k * sharedSize + sid] = 0.0;
        }
    }

    if (tc == 0 && tr == 0) {
        int sid = (ts_r - HALO) * sharedWidth + (ts_c - HALO);
        
        for (int k = 0; k < NUMBER_OF_OUTFLOWS; k++) {
            f_shared[k * sharedSize + sid] = 0.0;
        }
    }

    if (tc == blockDim.x - 1 && tr == 0) {
        int sid = (ts_r - HALO) * sharedWidth + (ts_c + HALO);
        
        for (int k = 0; k < NUMBER_OF_OUTFLOWS; k++) {
            f_shared[k * sharedSize + sid] = 0.0;
        }
    }

    if (tc == 0 && tr == blockDim.y - 1) {
        int sid = (ts_r + HALO) * sharedWidth + (ts_c - HALO);
        
        for (int k = 0; k < NUMBER_OF_OUTFLOWS; k++) {
            f_shared[k * sharedSize + sid] = 0.0;
        }
    }

    if (tc == blockDim.x - 1 && tr == blockDim.y - 1) {
        int sid = (ts_r + HALO) * sharedWidth + (ts_c + HALO);
        
        for (int k = 0; k < NUMBER_OF_OUTFLOWS; k++) {
            f_shared[k * sharedSize + sid] = 0.0;
        }
    }

    __syncthreads();

    double rad = sqrt(2.0);

    
    if (i < rows && j < cols) {
        double h0 = sh[idx];
        
        if (h0 > 0.0) {
            bool eliminated[MOORE_NEIGHBORS];
            double z[MOORE_NEIGHBORS];
            double h[MOORE_NEIGHBORS];
            double H[MOORE_NEIGHBORS];
            double theta[MOORE_NEIGHBORS];


            double sz0 = sz[idx];
            double T_val = st[idx];

            double rr = pow(10.0, d_a + d_b * T_val);
            double hc = pow(10.0, d_c + d_d * T_val);
              double w= d_pc;
  double pr= rr;

            for (int k = 0; k < MOORE_NEIGHBORS; k++) {
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
                } else {
                    h[k] = 0.0;
                    z[k] = sz0;
                }
            }

            H[0] = z[0];
            theta[0] = 0.0;
            eliminated[0] = false;

            for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                eliminated[k] = true;
                H[k] = 0.0;
                theta[k] = 0.0;
                if (z[0] + h[0] > z[k] + h[k]) {
                    H[k] = z[k] + h[k];
                    theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w);
                    eliminated[k] = false;
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
                    f_shared[outflow_idx * sharedSize + tid_s] = pr * (avg - H[k]);
                }
            }
        }
    }

    
    // Halo sinistro
    if (tc == 0) {
        int gi = i;
        int gj = j - HALO;
        int sid = ts_r * sharedWidth + (ts_c - HALO);

        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            double h0 = sh[gidx];

            if (h0 > 0.0) {
                bool eliminated[MOORE_NEIGHBORS];
                double z[MOORE_NEIGHBORS];
                double h[MOORE_NEIGHBORS];
                double H[MOORE_NEIGHBORS];
                double theta[MOORE_NEIGHBORS];

                double sz0 = sz[gidx];
                double T_val = st[gidx];

                double rr = pow(10.0, d_a + d_b * T_val);
                double hc = pow(10.0, d_c + d_d * T_val);

                  double w= d_pc;
  double pr= rr;

                for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                    int ni = gi + d_Xi[k];
                    int nj = gj + d_Xj[k];

                    bool is_valid = (ni >= 0 && ni < rows && nj >= 0 && nj < cols);

                    if (is_valid) {
                        int idx_k = ni * cols + nj;
                        double sz_k = sz[idx_k];
                        h[k] = sh[idx_k];

                        if (k < VON_NEUMANN_NEIGHBORS)
                            z[k] = sz_k;
                        else
                            z[k] = sz0 - (sz0 - sz_k) / rad;
                    } else {
                        h[k] = 0.0;
                        z[k] = sz0;
                    }
                }

                H[0] = z[0];
                theta[0] = 0.0;
                eliminated[0] = false;

                for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                                            eliminated[k] = true;
                        H[k] = 0.0;
                        theta[k] = 0.0;
                    if (z[0] + h[0] > z[k] + h[k]) {
                        H[k] = z[k] + h[k];
                        theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w);
                        eliminated[k] = false;
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
                        f_shared[outflow_idx * sharedSize + sid] = pr * (avg - H[k]);
                    }
                }
            }
        }
    }

    // Halo destro
    if (tc == blockDim.x - 1) {
        int gi = i;
        int gj = j + HALO;
        int sid = ts_r * sharedWidth + (ts_c + HALO);

        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            double h0 = sh[gidx];

            if (h0 > 0.0) {
                bool eliminated[MOORE_NEIGHBORS];
                double z[MOORE_NEIGHBORS];
                double h[MOORE_NEIGHBORS];
                double H[MOORE_NEIGHBORS];
                double theta[MOORE_NEIGHBORS];


                double sz0 = sz[gidx];
                double T_val = st[gidx];

                double rr = pow(10.0, d_a + d_b * T_val);
                double hc = pow(10.0, d_c + d_d * T_val);
                  double w= d_pc;
  double pr= rr;

                for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                    int ni = gi + d_Xi[k];
                    int nj = gj + d_Xj[k];

                    bool is_valid = (ni >= 0 && ni < rows && nj >= 0 && nj < cols);

                    if (is_valid) {
                        int idx_k = ni * cols + nj;
                        double sz_k = sz[idx_k];
                        h[k] = sh[idx_k];

                        if (k < VON_NEUMANN_NEIGHBORS)
                            z[k] = sz_k;
                        else
                            z[k] = sz0 - (sz0 - sz_k) / rad;
                    } else {
                        h[k] = 0.0;
                        z[k] = sz0;
                    }
                }

                H[0] = z[0];
                theta[0] = 0.0;
                eliminated[0] = false;

                for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                                            eliminated[k] = true;
                        H[k] = 0.0;
                        theta[k] = 0.0;
                    if (z[0] + h[0] > z[k] + h[k]) {
                        H[k] = z[k] + h[k];
                        theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w);
                        eliminated[k] = false;
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
                        f_shared[outflow_idx * sharedSize + sid] = pr * (avg - H[k]);
                    }
                }
            }
        }
    }

    // Halo superiore
    if (tr == 0) {
        int gi = i - HALO;
        int gj = j;
        int sid = (ts_r - HALO) * sharedWidth + ts_c;

        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            double h0 = sh[gidx];

            if (h0 > 0.0) {
                bool eliminated[MOORE_NEIGHBORS];
                double z[MOORE_NEIGHBORS];
                double h[MOORE_NEIGHBORS];
                double H[MOORE_NEIGHBORS];
                double theta[MOORE_NEIGHBORS];

                double sz0 = sz[gidx];
                double T_val = st[gidx];

                double rr = pow(10.0, d_a + d_b * T_val);
                double hc = pow(10.0, d_c + d_d * T_val);
                  double w= d_pc;
  double pr= rr;

                for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                    int ni = gi + d_Xi[k];
                    int nj = gj + d_Xj[k];

                    bool is_valid = (ni >= 0 && ni < rows && nj >= 0 && nj < cols);

                    if (is_valid) {
                        int idx_k = ni * cols + nj;
                        double sz_k = sz[idx_k];
                        h[k] = sh[idx_k];

                        if (k < VON_NEUMANN_NEIGHBORS)
                            z[k] = sz_k;
                        else
                            z[k] = sz0 - (sz0 - sz_k) / rad;
                    } else {
                        h[k] = 0.0;
                        z[k] = sz0;
                    }
                }

                H[0] = z[0];
                theta[0] = 0.0;
                eliminated[0] = false;

                for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                                            eliminated[k] = true;
                        H[k] = 0.0;
                        theta[k] = 0.0;
                    if (z[0] + h[0] > z[k] + h[k]) {
                        H[k] = z[k] + h[k];
                        theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w);
                        eliminated[k] = false;
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
                        f_shared[outflow_idx * sharedSize + sid] = pr * (avg - H[k]);
                    }
                }
            }
        }
    }

    // Halo inferiore
    if (tr == blockDim.y - 1) {
        int gi = i + HALO;
        int gj = j;
        int sid = (ts_r + HALO) * sharedWidth + ts_c;

        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            double h0 = sh[gidx];

            if (h0 > 0.0) {
                bool eliminated[MOORE_NEIGHBORS];
                double z[MOORE_NEIGHBORS];
                double h[MOORE_NEIGHBORS];
                double H[MOORE_NEIGHBORS];
                double theta[MOORE_NEIGHBORS];

                double sz0 = sz[gidx];
                double T_val = st[gidx];

                double rr = pow(10.0, d_a + d_b * T_val);
                double hc = pow(10.0, d_c + d_d * T_val);

                  double w= d_pc;
  double pr= rr;

                for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                    int ni = gi + d_Xi[k];
                    int nj = gj + d_Xj[k];

                    bool is_valid = (ni >= 0 && ni < rows && nj >= 0 && nj < cols);

                    if (is_valid) {
                        int idx_k = ni * cols + nj;
                        double sz_k = sz[idx_k];
                        h[k] = sh[idx_k];

                        if (k < VON_NEUMANN_NEIGHBORS)
                            z[k] = sz_k;
                        else
                            z[k] = sz0 - (sz0 - sz_k) / rad;
                    } else {
                        h[k] = 0.0;
                        z[k] = sz0;
                    }
                }

                H[0] = z[0];
                theta[0] = 0.0;
                eliminated[0] = false;

                for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                        eliminated[k] = true;
                        H[k] = 0.0;
                        theta[k] = 0.0;
                    if (z[0] + h[0] > z[k] + h[k]) {
                        H[k] = z[k] + h[k];
                        theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w);
                        eliminated[k] = false;
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
                        f_shared[outflow_idx * sharedSize + sid] = pr * (avg - H[k]);
                    }
                }
            }
        }
    }

    // Angolo top-left
    if (tc == 0 && tr == 0) {
        int gi = i - HALO;
        int gj = j - HALO;
        int sid = (ts_r - HALO) * sharedWidth + (ts_c - HALO);

        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            double h0 = sh[gidx];

            if (h0 > 0.0) {
                bool eliminated[MOORE_NEIGHBORS];
                double z[MOORE_NEIGHBORS];
                double h[MOORE_NEIGHBORS];
                double H[MOORE_NEIGHBORS];
                double theta[MOORE_NEIGHBORS];

                double sz0 = sz[gidx];
                double T_val = st[gidx];

                double rr = pow(10.0, d_a + d_b * T_val);
                double hc = pow(10.0, d_c + d_d * T_val);

                  double w= d_pc;
  double pr= rr;

                for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                    int ni = gi + d_Xi[k];
                    int nj = gj + d_Xj[k];

                    bool is_valid = (ni >= 0 && ni < rows && nj >= 0 && nj < cols);

                    if (is_valid) {
                        int idx_k = ni * cols + nj;
                        double sz_k = sz[idx_k];
                        h[k] = sh[idx_k];

                        if (k < VON_NEUMANN_NEIGHBORS)
                            z[k] = sz_k;
                        else
                            z[k] = sz0 - (sz0 - sz_k) / rad;
                    } else {
                        h[k] = 0.0;
                        z[k] = sz0;
                    }

                }

                H[0] = z[0];
                theta[0] = 0.0;
                eliminated[0] = false;

                for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                                            eliminated[k] = true;
                        H[k] = 0.0;
                        theta[k] = 0.0;
                    if (z[0] + h[0] > z[k] + h[k]) {
                        H[k] = z[k] + h[k];
                        theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w);
                        eliminated[k] = false;
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
                        f_shared[outflow_idx * sharedSize + sid] = pr * (avg - H[k]);
                    }
                }
            }
        }
    }

    // Angolo top-right
    if (tc == blockDim.x - 1 && tr == 0) {
        int gi = i - HALO;
        int gj = j + HALO;
        int sid = (ts_r - HALO) * sharedWidth + (ts_c + HALO);

        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            double h0 = sh[gidx];

            if (h0 > 0.0) {
                bool eliminated[MOORE_NEIGHBORS];
                double z[MOORE_NEIGHBORS];
                double h[MOORE_NEIGHBORS];
                double H[MOORE_NEIGHBORS];
                double theta[MOORE_NEIGHBORS];

                double sz0 = sz[gidx];
                double T_val = st[gidx];

                double rr = pow(10.0, d_a + d_b * T_val);
                double hc = pow(10.0, d_c + d_d * T_val);

                  double w= d_pc;
  double pr= rr;

                for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                    int ni = gi + d_Xi[k];
                    int nj = gj + d_Xj[k];

                    bool is_valid = (ni >= 0 && ni < rows && nj >= 0 && nj < cols);

                    if (is_valid) {
                        int idx_k = ni * cols + nj;
                        double sz_k = sz[idx_k];
                        h[k] = sh[idx_k];

                        if (k < VON_NEUMANN_NEIGHBORS)
                            z[k] = sz_k;
                        else
                            z[k] = sz0 - (sz0 - sz_k) / rad;
                    } else {
                        h[k] = 0.0;
                        z[k] = sz0;
                    }

                }

                H[0] = z[0];
                theta[0] = 0.0;
                eliminated[0] = false;

                for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                                            eliminated[k] = true;
                        H[k] = 0.0;
                        theta[k] = 0.0;
                    if (z[0] + h[0] > z[k] + h[k]) {
                        H[k] = z[k] + h[k];
                        theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w);
                        eliminated[k] = false;
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
                        f_shared[outflow_idx * sharedSize + sid] = pr * (avg - H[k]);
                    }
                }
            }
        }
    }

    // Angolo bottom-left
    if (tc == 0 && tr == blockDim.y - 1) {
        int gi = i + HALO;
        int gj = j - HALO;
        int sid = (ts_r + HALO) * sharedWidth + (ts_c - HALO);

        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            double h0 = sh[gidx];

            if (h0 > 0.0) {
                bool eliminated[MOORE_NEIGHBORS];
                double z[MOORE_NEIGHBORS];
                double h[MOORE_NEIGHBORS];
                double H[MOORE_NEIGHBORS];
                double theta[MOORE_NEIGHBORS];


                double sz0 = sz[gidx];
                double T_val = st[gidx];

                double rr = pow(10.0, d_a + d_b * T_val);
                double hc = pow(10.0, d_c + d_d * T_val);

                  double w= d_pc;
  double pr= rr;

                for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                    int ni = gi + d_Xi[k];
                    int nj = gj + d_Xj[k];

                    bool is_valid = (ni >= 0 && ni < rows && nj >= 0 && nj < cols);

                    if (is_valid) {
                        int idx_k = ni * cols + nj;
                        double sz_k = sz[idx_k];
                        h[k] = sh[idx_k];

                        if (k < VON_NEUMANN_NEIGHBORS)
                            z[k] = sz_k;
                        else
                            z[k] = sz0 - (sz0 - sz_k) / rad;
                    } else {
                        h[k] = 0.0;
                        z[k] = sz0;
                    }

                }

                H[0] = z[0];
                theta[0] = 0.0;
                eliminated[0] = false;

                for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                                            eliminated[k] = true;
                        H[k] = 0.0;
                        theta[k] = 0.0;
                    if (z[0] + h[0] > z[k] + h[k]) {
                        H[k] = z[k] + h[k];
                        theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w);
                        eliminated[k] = false;
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
                        f_shared[outflow_idx * sharedSize + sid] = pr * (avg - H[k]);
                    }
                }
            }
        }
    }

    // Angolo bottom-right
    if (tc == blockDim.x - 1 && tr == blockDim.y - 1) {
        int gi = i + HALO;
        int gj = j + HALO;
        int sid = (ts_r + HALO) * sharedWidth + (ts_c + HALO);

        if (gi >= 0 && gi < rows && gj >= 0 && gj < cols) {
            int gidx = gi * cols + gj;
            double h0 = sh[gidx];

            if (h0 > 0.0) {
                bool eliminated[MOORE_NEIGHBORS];
                double z[MOORE_NEIGHBORS];
                double h[MOORE_NEIGHBORS];
                double H[MOORE_NEIGHBORS];
                double theta[MOORE_NEIGHBORS];

                double sz0 = sz[gidx];
                double T_val = st[gidx];

                double rr = pow(10.0, d_a + d_b * T_val);
                double hc = pow(10.0, d_c + d_d * T_val);

                  double w= d_pc;
  double pr= rr;

                for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                    int ni = gi + d_Xi[k];
                    int nj = gj + d_Xj[k];

                    bool is_valid = (ni >= 0 && ni < rows && nj >= 0 && nj < cols);

                    if (is_valid) {
                        int idx_k = ni * cols + nj;
                        double sz_k = sz[idx_k];
                        h[k] = sh[idx_k];

                        if (k < VON_NEUMANN_NEIGHBORS)
                            z[k] = sz_k;
                        else
                            z[k] = sz0 - (sz0 - sz_k) / rad;
                    } else {
                        h[k] = 0.0;
                        z[k] = sz0;
                    }

                }

                H[0] = z[0];
                theta[0] = 0.0;
                eliminated[0] = false;

                for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                                            eliminated[k] = true;
                        H[k] = 0.0;
                        theta[k] = 0.0;
                    if (z[0] + h[0] > z[k] + h[k]) {
                        H[k] = z[k] + h[k];
                        theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w);
                        eliminated[k] = false;
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
                        f_shared[outflow_idx * sharedSize + sid] = pr * (avg - H[k]);
                    }
                }
            }
        }
    }

   __syncthreads();

    if (i >= rows || j >= cols) return;

    const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};

    double initial_h = sh[idx];
    double initial_t = st[idx];

    double h_next_val = initial_h;
    double t_next_val = initial_h * initial_t;

    for (int n = 1; n < MOORE_NEIGHBORS; n++) {
        int ni = i + d_Xi[n];
        int nj = j + d_Xj[n];

        if (ni < 0 || ni >= rows || nj < 0 || nj >= cols)
            continue;

        int ts_r_n = ts_r + d_Xi[n];
        int ts_c_n = ts_c + d_Xj[n];
        int tid_s_n = ts_r_n * sharedWidth + ts_c_n;

        int out_layer = n - 1;
        double outFlow = f_shared[out_layer * sharedSize + tid_s];

        int in_layer = inflowsIndices[n - 1];
        double inFlow = f_shared[in_layer * sharedSize + tid_s_n];

        double neigh_t = st[ni * cols + nj];

        h_next_val += (inFlow - outFlow);
        t_next_val += (inFlow * neigh_t - outFlow * initial_t);
    }

    if (h_next_val > 0) {
        t_next_val /= h_next_val;
        st_next[idx] = t_next_val;
        sh_next[idx] = h_next_val;
    }
}