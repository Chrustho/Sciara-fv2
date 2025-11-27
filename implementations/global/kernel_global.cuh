#pragma once

#include "../../src/vent.h"
#include "../../src/Sciara.h"

void emitLava(
    int i,
    int j,
    Sciara *sciara);

__global__ void computeOutflows_Global(
    Sciara *sciara);


__global__ void massBalance_Global(
    Sciara *sciara);  


__global__ void computeNewTemperatureAndSolidification_Global(
    Sciara *sciara         
);

__global__ void boundaryConditions_Global(
    Sciara *sciara     
);

__global__ void reduceAdd_Kernel(
    int n, const double *buffer, double *global_result);


