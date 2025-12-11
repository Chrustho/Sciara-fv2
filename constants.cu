#include <cuda_runtime.h>

// Definizioni uniche - solo qui!
__constant__ int rows;
__constant__ int cols;
__constant__ double c_pc;
__constant__ double c_a;
__constant__ double c_b;
__constant__ double c_c;
__constant__ double c_d;
__constant__ double c_pepsilon;
__constant__ double c_psigma;
__constant__ double c_pclock;
__constant__ double c_pcool;
__constant__ double c_prho;
__constant__ double c_pcv;
__constant__ double c_pac;
__constant__ double c_ptsol;