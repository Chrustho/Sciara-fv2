#pragma once

#include "../../src/vent.h"
#include "../../src/Sciara.h"


__global__ void computeOutflows_Tiled(
    Sciara *sciara, const unsigned int tileX, const int tileY);


__global__ void massBalance_Tiled(
    Sciara *sciara);  


__global__ void computeNewTemperatureAndSolidification_Tiled(
    Sciara *sciara         
);

__global__ void boundaryConditions_Tiled(
    Sciara *sciara     
);


