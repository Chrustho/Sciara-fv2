#pragma once
#include <cuda_runtime.h>

// Dichiarazioni extern - tutti i kernel le includono
extern __constant__ int rows;
extern __constant__ int cols;
extern __constant__ int d_Xi[9];
extern __constant__ int d_Xj[9];
extern __constant__ double d_pc;
extern __constant__ double d_a;
extern __constant__ double d_b;
extern __constant__ double d_c;
extern __constant__ double d_d;
extern __constant__ double d_pepsilon;
extern __constant__ double d_psigma;
extern __constant__ double d_pclock;
extern __constant__ double d_pcool;
extern __constant__ double d_prho;
extern __constant__ double d_pcv;
extern __constant__ double d_pac;
extern __constant__ double d_ptsol;
extern __constant__ double d_ptvent;
extern __constant__ double d_temp_factor;  
extern __constant__ double d_temp_divisor;


// Funzione per inizializzare le costanti (definita in constants.cu)
void initializeConstants(int rows, int cols,
                         double pc, double a, double b, double c, double d,
                         double pepsilon, double psigma, double pclock,
                         double pcool, double prho, double pcv, double pac,
                         double ptsol, double ptvent);