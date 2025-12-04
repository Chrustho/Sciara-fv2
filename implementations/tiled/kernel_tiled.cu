#pragma once

#include "../../src/vent.h"
#include "../../src/Sciara.h"
#include "../../implementations/tiled/kernel_tiled.cuh"



__global__ void computeOutflows_Tiled(
    Sciara *sciara, const unsigned int tileX, const int tileY){
    // Parametri del dominio
    int rows= sciara->domain->rows;
    int cols= sciara->domain->cols;

    // Vicini
    int *xi= sciara->X->Xi;
    int *xj=sciara->X->Xj;

    // Buffers
    double *sh=sciara->substates->Sh;
    double *st=sciara->substates->ST;
    double *sz= sciara->substates->Sz;

    // a, b, c, d
    double _a= sciara->parameters->a;
    double _b=sciara->parameters->b;
    double _c=sciara->parameters->c;
    double _d=sciara->parameters->d;

    //taglia della shared dim(sz)*2 (sto considerando Sh,st)
    //rimane mf fuori

    extern __shared__ double shared_mem[];
    
    // riga 1: sh
    // riga 2: st
    // riga 3: sz
    int tc= threadIdx.x;
    int tr= threadIdx.y;

    int buffer_size= rows * cols *sizeof(double);

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= rows || j >= cols) return;

    int idx = i * cols + j;
    int tid= tr* tileX +tc;

    if (i <rows && j < cols)
    {
        // carico Sh
        shared_mem[tid] = sh[idx];
        //carico st
        
    }

    __syncthreads();
    



}


__global__ void massBalance_Tiled(
    Sciara *sciara);  


__global__ void computeNewTemperatureAndSolidification_Tiled(
    Sciara *sciara         
);

__global__ void boundaryConditions_Tiled(
    Sciara *sciara     
);

