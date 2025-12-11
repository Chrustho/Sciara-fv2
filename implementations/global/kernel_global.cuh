#pragma once

#include "../../src/vent.h"
#include "../../src/Sciara.h"


__global__ void computeOutflows_Global(
    double *sh, double *st, double *sz, double *mf);


__global__ void massBalance_Global(
    double *sh, double *sh_next, double *st, double *st_next, double *mf);  


__global__ void computeNewTemperatureAndSolidification_Global(
    double* __restrict__ sh, double*  __restrict__ sh_next, double* __restrict__ st, double* __restrict__ st_next, double* __restrict__ sz, double* __restrict__ sz_next, double* __restrict__ mhs, const bool *__restrict__ mb         
);

__global__ void boundaryConditions_Global(
    Sciara *sciara     
);
