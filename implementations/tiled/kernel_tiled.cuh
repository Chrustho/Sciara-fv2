#pragma once

#include "../../src/vent.h"
#include "../../src/Sciara.h"


__global__ void computeOutflows_Tiled(
        double *sh, double *st, double *sz, double *mf);


__global__ void massBalance_Tiled(
    double *sh, double *sh_next, double *st, double *st_next, double *mf);  


__global__ void computeNewTemperatureAndSolidification_Tiled();

__global__ void boundaryConditions_Tiled();


