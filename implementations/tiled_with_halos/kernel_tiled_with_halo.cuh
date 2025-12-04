#pragma once

#include "../../src/vent.h"
#include "../../src/Sciara.h"


__global__ void computeOutflows_Tiled_wH(
    Sciara *sciara);


__global__ void massBalance_Tiled_wH(
    Sciara *sciara);  


__global__ void computeNewTemperatureAndSolidification_Tiled_wH();

__global__ void boundaryConditions_Tiled_wH();