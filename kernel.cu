//TODO: sostituire tutti questi parametri con delle struct
__global__ void emitLava_Global(
    int r, 
    int c, 
    TVent *vents,    //FIXME      
    int num_vents,        
    double elapsed_time, 
    double Pclock, 
    double emission_time, 
    double *total_emitted_lava, 
    double Pac, 
    double PTvent, 
    double *Sh, 
    double *Sh_next, 
    double *ST_next
);

__global__ void computeOutflows_Global(
    int r, 
    int c, 
    int *Xi, 
    int *Xj, 
    double *Sz, 
    double *Sh, 
    double *ST, 
    double *Mf, 
    double Pc, 
    double _a, 
    double _b, 
    double _c, 
    double _d
);

__global__ void computeOutflows_Tiled(
    int r, int c, int *Xi, int *Xj, double *Sz, double *Sh, double *ST, double *Mf, 
    double Pc, double _a, double _b, double _c, double _d
);

__global__ void massBalance_Global(
    int r, 
    int c, 
    int *Xi, 
    int *Xj, 
    double *Sh, 
    double *Sh_next, 
    double *ST, 
    double *ST_next, 
    double *Mf
);

// Versione Tiled (Shared Memory)
__global__ void massBalance_Tiled(
    int r, int c, int *Xi, int *Xj, double *Sh, double *Sh_next, double *ST, double *ST_next, double *Mf
);

__global__ void computeNewTemperatureAndSolidification_Global(
    int r, 
    int c, 
    double Pepsilon, 
    double Psigma, 
    double Pclock, 
    double Pcool, 
    double Prho, 
    double Pcv, 
    double Pac, 
    double PTsol, 
    double *Sz, 
    double *Sz_next, 
    double *Sh, 
    double *Sh_next, 
    double *ST, 
    double *ST_next, 
    double *Mf, 
    double *Mhs, 
    bool *Mb
);

__global__ void boundaryConditions_Global(
    int r, 
    int c, 
    double *Mf, 
    bool *Mb, 
    double *Sh, 
    double *Sh_next, 
    double *ST, 
    double *ST_next
);

__global__ void reduceAdd_Global(
    int r, 
    int c, 
    double *buffer,      
    double *partial_sums
);

