############
# COMPILER #
############

NVCC      = nvcc
# Aggiunto -I./implementations/global per trovare gli header dei kernel
INCLUDES  = -I./src -I./implementations/global
NVCODE    = -gencode arch=compute_52,code=sm_52 -ftz=true
NVFLAGS   = -O2 -std=c++14 $(NVCODE) $(INCLUDES)

###########
# SOURCES #
###########

# 1. File C++ nella root (es. Sciara.cpp)
SRCS_ROOT = $(wildcard *.cpp)
# 2. File C++ nella cartella src
SRCS_SRC  = $(wildcard src/*.cpp)
# 3. File CUDA dei kernel (FONDAMENTALE: mancava questo!)
SRCS_KERNELS = implementations/global/kernel_global.cu
# 4. File CUDA principale
SRCS_MAIN = sciara_fv2.cu Sciara.cu

# Uniamo tutto in una lista
ALL_SOURCES = $(SRCS_ROOT) $(SRCS_SRC) $(SRCS_KERNELS) $(SRCS_MAIN) 

###########
# DATASET #
###########

INPUT_CONFIG=./data/2006/2006_000000000000.cfg
OUTPUT_CONFIG=./data/2006/output_2006
OUTPUT=./data/2006/output_2006_000000016000_Temperature.asc #md5sum: 0c071cd864046d3c6aaf30997290ad6c
STEPS=1000
REDUCE_INTERVL=1000
THICKNESS_THRESHOLD=1.0

###############
# COMPILATION #
###############

EXEC_CUDA = sciara_cuda

default: all

all: $(EXEC_CUDA)

$(EXEC_CUDA):
	$(NVCC) $(NVFLAGS) $(ALL_SOURCES) -o $(EXEC_CUDA)

#############
# EXECUTION #
#############

run_cuda: $(EXEC_CUDA)
	./$(EXEC_CUDA) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

############
# CLEAN UP #
############

clean:
	rm -f $(EXEC_CUDA) *.o *output*

wipe:
	rm -f *.o *output*
