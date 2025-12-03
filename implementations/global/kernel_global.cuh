#pragma once

#include "../../src/vent.h"
#include "../../src/Sciara.h"


__global__ void emitLava_Global(
    Sciara *sciara, 
    GPUVent *d_vents,      // Usiamo la struct semplificata
    int num_vents, 
    double *d_total_emitted_lava
);
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


