#include "constants.cuh"
#include "src/Sciara.h"

__constant__ int rows;
__constant__ int cols;
__constant__ int d_Xi[9];
__constant__ int d_Xj[9];
__constant__ double d_pc;
__constant__ double d_a;
__constant__ double d_b;
__constant__ double d_c;
__constant__ double d_d;
__constant__ double d_pepsilon;
__constant__ double d_psigma;
__constant__ double d_pclock;
__constant__ double d_pcool;
__constant__ double d_prho;
__constant__ double d_pcv;
__constant__ double d_pac;
__constant__ double d_ptsol;
__constant__ double d_ptvent;
__constant__ double d_temp_factor;  
__constant__ double d_temp_divisor;

static const int h_Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1};
static const int h_Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1};

void initializeConstants(Sciara* sciara) {
    int hrows = sciara->domain->rows;
    int hcols = sciara->domain->cols;
    
    cudaMemcpyToSymbol(rows, &hrows, sizeof(int));
    cudaMemcpyToSymbol(cols, &hcols, sizeof(int));
    cudaMemcpyToSymbol(d_Xi, h_Xi, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_Xj, h_Xj, 9 * sizeof(int));
    
    cudaMemcpyToSymbol(d_pc, &sciara->parameters->Pc, sizeof(double));
    cudaMemcpyToSymbol(d_a, &sciara->parameters->a, sizeof(double));
    cudaMemcpyToSymbol(d_b, &sciara->parameters->b, sizeof(double));
    cudaMemcpyToSymbol(d_c, &sciara->parameters->c, sizeof(double));
    cudaMemcpyToSymbol(d_d, &sciara->parameters->d, sizeof(double));
    cudaMemcpyToSymbol(d_pepsilon, &sciara->parameters->Pepsilon, sizeof(double));
    cudaMemcpyToSymbol(d_psigma, &sciara->parameters->Psigma, sizeof(double));
    cudaMemcpyToSymbol(d_pclock, &sciara->parameters->Pclock, sizeof(double));
    cudaMemcpyToSymbol(d_pcool, &sciara->parameters->Pcool, sizeof(double));
    cudaMemcpyToSymbol(d_prho, &sciara->parameters->Prho, sizeof(double));
    cudaMemcpyToSymbol(d_pcv, &sciara->parameters->Pcv, sizeof(double));
    cudaMemcpyToSymbol(d_pac, &sciara->parameters->Pac, sizeof(double));
    cudaMemcpyToSymbol(d_ptsol, &sciara->parameters->PTsol, sizeof(double));
    cudaMemcpyToSymbol(d_ptvent, &sciara->parameters->PTvent, sizeof(double));

    double temp_factor = 3.0 * sciara->parameters->Pepsilon * 
                          sciara->parameters->Psigma * 
                          sciara->parameters->Pclock * 
                          sciara->parameters->Pcool;
cudaMemcpyToSymbol(d_temp_factor, &temp_factor, sizeof(double));

double temp_divisor = sciara->parameters->Prho * 
                      sciara->parameters->Pcv * 
                      sciara->parameters->Pac;
cudaMemcpyToSymbol(d_temp_divisor, &temp_divisor, sizeof(double));
    
    cudaDeviceSynchronize();
}