#include "../../src/vent.h"
#include "../../src/Sciara.h"
#include "../../implementations/tiled_with_halos/kernel_tiled_with_halo.cuh"

#ifndef HALO
#define HALO 1
#endif

__constant__ int d_Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1}; 
__constant__ int d_Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1}; 


__constant__ int rows  = 378;
__constant__ int cols = 517;

// Dimensioni del dominio (da impostare prima del lancio o passare come parametri)
__constant__ int d_rows = 378;
__constant__ int d_cols = 517;

/**
 * CfAMo Kernel - Conflict-free Accumulation Memory-optimized
 * 
 * Questo kernel fonde computeOutflows e massBalance in un unico kernel,
 * usando shared memory con halo cells per ottimizzare gli accessi.
 * 
 * Shared Memory Layout (5 buffer):
 *   - sh_s:     Altezza lava (input)
 *   - sz_s:     Quota terreno (input)
 *   - st_s:     Temperatura (input)
 *   - sh_accum: Accumulatore massa in ingresso
 *   - en_accum: Accumulatore energia in ingresso
 *
 * Dimensione shared memory richiesta: (blockDim.x + 2*HALO) * (blockDim.y + 2*HALO) * 5 * sizeof(double)
 */
__global__ void CfAMo_Kernel(Sciara *sciara) {

    // === ESTRAZIONE PUNTATORI E PARAMETRI ===
    double *sh = sciara->substates->Sh;
    double *st = sciara->substates->ST;
    double *sz = sciara->substates->Sz;
    
    double *sh_next = sciara->substates->Sh_next;
    double *st_next = sciara->substates->ST_next;

    double _a = sciara->parameters->a;
    double _b = sciara->parameters->b;
    double _c = sciara->parameters->c;
    double _d = sciara->parameters->d;
    double pc = sciara->parameters->Pc;

    // === CALCOLO DIMENSIONI SHARED MEMORY ===
    const int sharedWidth = blockDim.x + 2 * HALO;   // es: 16 + 2 = 18
    const int sharedHeight = blockDim.y + 2 * HALO;  // es: 16 + 2 = 18
    const int sharedSize = sharedWidth * sharedHeight; // es: 324

    // === SETUP SHARED MEMORY ===
    extern __shared__ double shared_mem[];
    
    double *sh_s = shared_mem;                        // [0, sharedSize)
    double *sz_s = shared_mem + sharedSize;           // [sharedSize, 2*sharedSize)
    double *st_s = shared_mem + 2 * sharedSize;       // [2*sharedSize, 3*sharedSize)
    double *sh_accum = shared_mem + 3 * sharedSize;   // [3*sharedSize, 4*sharedSize)
    double *en_accum = shared_mem + 4 * sharedSize;   // [4*sharedSize, 5*sharedSize)

    // === CALCOLO INDICI ===
    // Indici locali del thread nel blocco
    const int tc = threadIdx.x;
    const int tr = threadIdx.y;
    
    // Indici nella shared memory (con offset per halo)
    const int tc_s = tc + HALO;
    const int tr_s = tr + HALO;
    const int tid_s = tr_s * sharedWidth + tc_s;

    // Indici globali nel dominio
    const int j = blockIdx.x * blockDim.x + tc;
    const int i = blockIdx.y * blockDim.y + tr;
    const int idx = i * d_cols + j;

    // Validità della cella nel dominio di simulazione
    const bool is_valid_cell = (i >= 0 && i < d_rows && j >= 0 && j < d_cols);

    // =======================================================
    // FASE 1: CARICAMENTO DATI IN SHARED MEMORY
    // =======================================================
    
    // 1.1: Ogni thread carica la propria cella centrale
    if (is_valid_cell) {
        sh_s[tid_s] = sh[idx];
        sz_s[tid_s] = sz[idx];
        st_s[tid_s] = st[idx];
    } else {
        sh_s[tid_s] = 0.0;
        sz_s[tid_s] = 0.0;
        st_s[tid_s] = 0.0;
    }

    // 1.2: Caricamento celle HALO (bordi della tile)
    
    // Halo SINISTRO (colonna sinistra del blocco carica cella a sinistra)
    if (tc == 0) {
        int halo_j = j - HALO;
        int halo_idx_s = tr_s * sharedWidth + (tc_s - HALO);
        if (halo_j >= 0 && halo_j < d_cols && i >= 0 && i < d_rows) {
            int halo_idx_g = i * d_cols + halo_j;
            sh_s[halo_idx_s] = sh[halo_idx_g];
            sz_s[halo_idx_s] = sz[halo_idx_g];
            st_s[halo_idx_s] = st[halo_idx_g];
        } else {
            sh_s[halo_idx_s] = 0.0;
            sz_s[halo_idx_s] = 0.0;
            st_s[halo_idx_s] = 0.0;
        }
    }
    
    // Halo DESTRO (colonna destra del blocco carica cella a destra)
    if (tc == blockDim.x - 1) {
        int halo_j = j + HALO;
        int halo_idx_s = tr_s * sharedWidth + (tc_s + HALO);
        if (halo_j >= 0 && halo_j < d_cols && i >= 0 && i < d_rows) {
            int halo_idx_g = i * d_cols + halo_j;
            sh_s[halo_idx_s] = sh[halo_idx_g];
            sz_s[halo_idx_s] = sz[halo_idx_g];
            st_s[halo_idx_s] = st[halo_idx_g];
        } else {
            sh_s[halo_idx_s] = 0.0;
            sz_s[halo_idx_s] = 0.0;
            st_s[halo_idx_s] = 0.0;
        }
    }
    
    // Halo SUPERIORE (riga superiore del blocco carica cella sopra)
    if (tr == 0) {
        int halo_i = i - HALO;
        int halo_idx_s = (tr_s - HALO) * sharedWidth + tc_s;
        if (halo_i >= 0 && halo_i < d_rows && j >= 0 && j < d_cols) {
            int halo_idx_g = halo_i * d_cols + j;
            sh_s[halo_idx_s] = sh[halo_idx_g];
            sz_s[halo_idx_s] = sz[halo_idx_g];
            st_s[halo_idx_s] = st[halo_idx_g];
        } else {
            sh_s[halo_idx_s] = 0.0;
            sz_s[halo_idx_s] = 0.0;
            st_s[halo_idx_s] = 0.0;
        }
    }
    
    // Halo INFERIORE (riga inferiore del blocco carica cella sotto)
    if (tr == blockDim.y - 1) {
        int halo_i = i + HALO;
        int halo_idx_s = (tr_s + HALO) * sharedWidth + tc_s;
        if (halo_i >= 0 && halo_i < d_rows && j >= 0 && j < d_cols) {
            int halo_idx_g = halo_i * d_cols + j;
            sh_s[halo_idx_s] = sh[halo_idx_g];
            sz_s[halo_idx_s] = sz[halo_idx_g];
            st_s[halo_idx_s] = st[halo_idx_g];
        } else {
            sh_s[halo_idx_s] = 0.0;
            sz_s[halo_idx_s] = 0.0;
            st_s[halo_idx_s] = 0.0;
        }
    }
    
    // Halo ANGOLI (solo i 4 thread agli angoli del blocco)
    // Angolo TOP-LEFT
    if (tc == 0 && tr == 0) {
        int hi = i - HALO;
        int hj = j - HALO;
        int hidx_s = (tr_s - HALO) * sharedWidth + (tc_s - HALO);
        if (hi >= 0 && hi < d_rows && hj >= 0 && hj < d_cols) {
            int hidx_g = hi * d_cols + hj;
            sh_s[hidx_s] = sh[hidx_g];
            sz_s[hidx_s] = sz[hidx_g];
            st_s[hidx_s] = st[hidx_g];
        } else {
            sh_s[hidx_s] = 0.0;
            sz_s[hidx_s] = 0.0;
            st_s[hidx_s] = 0.0;
        }
    }
    
    // Angolo TOP-RIGHT
    if (tc == blockDim.x - 1 && tr == 0) {
        int hi = i - HALO;
        int hj = j + HALO;
        int hidx_s = (tr_s - HALO) * sharedWidth + (tc_s + HALO);
        if (hi >= 0 && hi < d_rows && hj >= 0 && hj < d_cols) {
            int hidx_g = hi * d_cols + hj;
            sh_s[hidx_s] = sh[hidx_g];
            sz_s[hidx_s] = sz[hidx_g];
            st_s[hidx_s] = st[hidx_g];
        } else {
            sh_s[hidx_s] = 0.0;
            sz_s[hidx_s] = 0.0;
            st_s[hidx_s] = 0.0;
        }
    }
    
    // Angolo BOTTOM-LEFT
    if (tc == 0 && tr == blockDim.y - 1) {
        int hi = i + HALO;
        int hj = j - HALO;
        int hidx_s = (tr_s + HALO) * sharedWidth + (tc_s - HALO);
        if (hi >= 0 && hi < d_rows && hj >= 0 && hj < d_cols) {
            int hidx_g = hi * d_cols + hj;
            sh_s[hidx_s] = sh[hidx_g];
            sz_s[hidx_s] = sz[hidx_g];
            st_s[hidx_s] = st[hidx_g];
        } else {
            sh_s[hidx_s] = 0.0;
            sz_s[hidx_s] = 0.0;
            st_s[hidx_s] = 0.0;
        }
    }
    
    // Angolo BOTTOM-RIGHT
    if (tc == blockDim.x - 1 && tr == blockDim.y - 1) {
        int hi = i + HALO;
        int hj = j + HALO;
        int hidx_s = (tr_s + HALO) * sharedWidth + (tc_s + HALO);
        if (hi >= 0 && hi < d_rows && hj >= 0 && hj < d_cols) {
            int hidx_g = hi * d_cols + hj;
            sh_s[hidx_s] = sh[hidx_g];
            sz_s[hidx_s] = sz[hidx_g];
            st_s[hidx_s] = st[hidx_g];
        } else {
            sh_s[hidx_s] = 0.0;
            sz_s[hidx_s] = 0.0;
            st_s[hidx_s] = 0.0;
        }
    }

    // SYNC 1: Tutti i dati in shared memory sono pronti
    __syncthreads();

    // =======================================================
    // FASE 2: INIZIALIZZAZIONE ACCUMULATORI E CALCOLO FLUSSI
    // =======================================================
    
    // Inizializza accumulatori a zero
    sh_accum[tid_s] = 0.0;
    en_accum[tid_s] = 0.0;
    
    // Array locale per i flussi calcolati (8 direzioni)
    double calculated_flows[8];
    for (int k = 0; k < 8; k++) {
        calculated_flows[k] = 0.0;
    }

    // Valori della cella centrale
    double h0 = 0.0;
    double t0 = 0.0;

    // Calcolo flussi solo se la cella è valida e ha lava
    if (is_valid_cell) {
        h0 = sh_s[tid_s];
        
        if (h0 > 0.0) {
            t0 = st_s[tid_s];
            double sz0 = sz_s[tid_s];
            
            // Array per l'algoritmo di minimizzazione
            bool eliminated[MOORE_NEIGHBORS];
            double z[MOORE_NEIGHBORS];
            double h[MOORE_NEIGHBORS];
            double H[MOORE_NEIGHBORS];
            double theta[MOORE_NEIGHBORS];
            double Pr[MOORE_NEIGHBORS];
            double w[MOORE_NEIGHBORS];

            // Calcolo parametri reologici
            double rr = pow(10.0, _a + _b * t0);
            double hc = pow(10.0, _c + _d * t0);
            double rad = sqrt(2.0);

            // Raccolta dati dei vicini dalla shared memory
            for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                // Calcola posizione del vicino in shared memory
                int ntc_s = tc_s + d_Xj[k];
                int ntr_s = tr_s + d_Xi[k];
                int nidx_s = ntr_s * sharedWidth + ntc_s;
                
                // Leggi dalla shared memory (sempre valido grazie all'halo)
                double h_k = sh_s[nidx_s];
                double sz_k = sz_s[nidx_s];
                
                h[k] = h_k;
                
                // Correzione altezza per celle diagonali
                if (k < VON_NEUMANN_NEIGHBORS) {
                    z[k] = sz_k;
                } else {
                    z[k] = sz0 - (sz0 - sz_k) / rad;
                }
                
                w[k] = pc;
                Pr[k] = rr;
            }

            // === ALGORITMO DI MINIMIZZAZIONE ===
            
            // Inizializzazione
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

            // Iterazione per trovare l'equilibrio
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
                
                if (counter != 0) {
                    avg /= (double)counter;
                }
                
                for (int k = 0; k < MOORE_NEIGHBORS; k++) {
                    if (!eliminated[k] && avg <= H[k]) {
                        eliminated[k] = true;
                        loop = true;
                    }
                }
            } while (loop);
            
            // Calcolo flussi finali
            for (int k = 1; k < MOORE_NEIGHBORS; k++) {
                if (!eliminated[k] && h[0] > hc * cos(theta[k])) {
                    calculated_flows[k - 1] = Pr[k] * (avg - H[k]);
                }
            }
        }
    }

    // SYNC 2: Tutti hanno calcolato i propri flussi
    __syncthreads();

    // =======================================================
    // FASE 3: ACCUMULO FLUSSI CON 8 STEP SINCRONIZZATI
    // =======================================================
    
    double total_outflow = 0.0;

    // 8 step, uno per ogni direzione del vicinato di Moore
    for (int step = 1; step < MOORE_NEIGHBORS; step++) {
        
        double my_flow = calculated_flows[step - 1];
        total_outflow += my_flow;

        // Calcola posizione del vicino destinatario in shared memory
        int tn_c_s = tc_s + d_Xj[step];
        int tn_r_s = tr_s + d_Xi[step];

        // Verifica che il vicino sia dentro la shared memory (incluso halo)
        if (tn_c_s >= 0 && tn_c_s < sharedWidth && 
            tn_r_s >= 0 && tn_r_s < sharedHeight) {
            
            int tid_s_neigh = tn_r_s * sharedWidth + tn_c_s;
            
            if (my_flow > 0.0) {
                // Accumula massa ed energia nel vicino
                sh_accum[tid_s_neigh] += my_flow;
                en_accum[tid_s_neigh] += my_flow * t0;
            }
        }
        
        // SYNC: Ogni step è sincronizzato per evitare race conditions
        __syncthreads();
    }

    // =======================================================
    // FASE 4: SCRITTURA RISULTATI IN MEMORIA GLOBALE
    // =======================================================
    
    // Solo celle valide nel dominio scrivono il risultato
    if (is_valid_cell) {
        
        double inflow_mass = sh_accum[tid_s];
        double inflow_energy = en_accum[tid_s];
        
        // Calcolo nuova altezza
        double h_new = h0 + inflow_mass - total_outflow;
        
        if (h_new > 0.0) {
            // Calcolo nuova temperatura (bilancio energetico)
            double e_residual = (h0 - total_outflow) * t0;
            double t_new = (e_residual + inflow_energy) / h_new;
            
            sh_next[idx] = h_new;
            st_next[idx] = t_new;
        } else {
            // Nessuna lava rimasta
            sh_next[idx] = 0.0;
            st_next[idx] = 0.0;
        }
    }
}

