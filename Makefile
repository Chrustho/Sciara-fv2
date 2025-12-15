############
# COMPILER #
############

ifndef CPPC
    CPPC=g++
endif

NVCC=nvcc
# Aggiungo -I. per assicurare che trovi gli header nella root corrente
NVFLAGS=-O3 -rdc=true -I. 

###########
# DATASET #
###########

INPUT_CONFIG=./data/2006/2006_000000000000.cfg
OUTPUT_CONFIG=./data/2006/output_2006
OUTPUT=./data/2006/output_2006_000000016000_Temperature.asc
STEPS=16000
REDUCE_INTERVL=1000
THICKNESS_THRESHOLD=1.0

###############################
# SOURCES & DEPENDENCIES
###############################

# 1. Helpers C++ (Input/Output, Utils)
CPP_HELPERS = $(wildcard src/*.cpp)

# 2. Helpers CUDA Comuni (Strutture dati, costanti, init)
CU_HELPERS = constants.cu Sciara.cu 

# 3. Kernel Implementations
# Definisco i percorsi per chiarezza
KERNEL_GLOBAL = implementations/global/kernel_global.cu

# Lista base di sorgenti per OGNI compilazione (C++ helpers + CUDA helpers)
BASE_SOURCES = $(CPP_HELPERS) $(CU_HELPERS)

###############################
# EXECUTABLES NAMING
###############################

EXEC_CUDA            = sciara_cuda
EXEC_CUDA_TILED      = sciara_cuda_tiled
EXEC_CUDA_TILED_HALO = sciara_cuda_tiled_halo
EXEC_CUDA_CFAME      = sciara_cuda_cfame
EXEC_CUDA_CFAMO      = sciara_cuda_cfamo

###############################
# TARGETS COMPILATION
###############################

default: all

all: sciara_cuda

# --- VERSIONE GLOBAL (BASE) ---
cuda:
	$(NVCC) $(NVFLAGS) sciara_fv2.cu $(KERNEL_GLOBAL) $(BASE_SOURCES) -o $(EXEC_CUDA)

# --- VERSIONE TILED ---
cuda_tiled:
	$(NVCC) $(NVFLAGS) sciara_fv2_tiled.cu $(KERNEL_GLOBAL) $(wildcard implementations/tiled/*.cu) $(BASE_SOURCES) -o $(EXEC_CUDA_TILED)

# --- VERSIONE TILED HALO ---
cuda_tiled_halo:
	$(NVCC) $(NVFLAGS) sciara_fv2_tiled_halo.cu $(KERNEL_GLOBAL) $(wildcard implementations/tiled_with_halos/*.cu) $(BASE_SOURCES) -o $(EXEC_CUDA_TILED_HALO)

# --- VERSIONE CFAME ---
cuda_cfame:
	$(NVCC) $(NVFLAGS) sciara_fv2_cfame.cu $(KERNEL_GLOBAL) $(wildcard implementations/cfame/*.cu) $(BASE_SOURCES) -o $(EXEC_CUDA_CFAME)

# --- VERSIONE CFAMO ---
cuda_cfamo:
	$(NVCC) $(NVFLAGS) sciara_fv2_cfamo.cu $(KERNEL_GLOBAL) $(wildcard implementations/cfamo/*.cu) $(BASE_SOURCES) -o $(EXEC_CUDA_CFAMO)

# Meta-target per compilare tutto
sciara_cuda: cuda cuda_tiled cuda_tiled_halo cuda_cfame cuda_cfamo

#############
# EXECUTION #
#############

THREADS = 8

# Esecuzione standard (usa la versione base CUDA)
run: run_cuda

run_cuda:
	./$(EXEC_CUDA) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run_cuda_tiled:
	./$(EXEC_CUDA_TILED) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run_cuda_tiled_halo:
	./$(EXEC_CUDA_TILED_HALO) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run_cuda_cfame:
	./$(EXEC_CUDA_CFAME) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run_cuda_cfamo:
	./$(EXEC_CUDA_CFAMO) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

############
#  PROFILE #
############

# Dipende da sciara_cuda (che compila tutto) o puoi mettere solo 'cuda' se profili solo quello
profile: sciara_cuda
	chmod +x run_profiling.sh
	./run_profiling.sh

############
# CLEAN UP #
############

clean:
	rm -f $(EXEC_CUDA) $(EXEC_CUDA_TILED) $(EXEC_CUDA_TILED_HALO) $(EXEC_CUDA_CFAME) $(EXEC_CUDA_CFAMO) *.o *out*

wipe:
	rm -f *.o *out*

clean-profile:
	rm -rf profiling_results/

clean-all: clean wipe clean-profile

.PHONY: default all cuda cuda_tiled cuda_tiled_halo cuda_cfame cuda_cfamo sciara_cuda run run_cuda run_cuda_tiled run_cuda_tiled_halo run_cuda_cfame run_cuda_cfamo profile clean wipe clean-profile clean-all
