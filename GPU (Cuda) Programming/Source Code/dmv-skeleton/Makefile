.PHONY: all clean query
.DEFAULT: all

DEBUG ?= 1
REGINFO ?= 1
GPU ?= 1
SERIAL ?= 1
OPENMP ?= 1
CXX ?= g++
CUDA_PREFIX = /usr/local/cuda
GPU_CXX ?= $(CUDA_PREFIX)/bin/nvcc
RM = /bin/rm

# Fermi GPUs in termis
nvcc_backend = g++
nvcc_emu = 
nvcc_arch = -arch sm_20

MAKE_CPPFLAGS = -D__FLOAT_VALUES -DEPS=1e-2
MAKE_CXXFLAGS = -c -Wall -fopenmp
MAKE_GPU_CPPFLAGS = -I$(CUDA_PREFIX)/include
MAKE_GPU_CXXFLAGS = -c -ccbin $(nvcc_backend) $(nvcc_emu) $(nvcc_arch)

ifeq ($(SERIAL), 1)
	MAKE_CPPFLAGS += -DSERIAL_KERNEL
else
    # we don't implement the serial version, so don't check
	MAKE_CPPFLAGS += -D_NOCHECK_
endif

ifeq ($(OPENMP), 1)
	MAKE_CPPFLAGS += -DOPENMP_KERNEL
endif

ifeq ($(GPU), 1)
	MAKE_CPPFLAGS += -DGPU_KERNEL
endif

ifeq ($(DEBUG), 1)
	MAKE_CPPFLAGS += -D_DEBUG_
	MAKE_CXXFLAGS += -g -O0
	MAKE_GPU_CXXFLAGS += -g -O0
else
	MAKE_CPPFLAGS += -DNDEBUG
	MAKE_CXXFLAGS += -O3
	MAKE_GPU_CXXFLAGS += -O3
endif

ifeq ($(REGINFO), 1)
	MAKE_GPU_CXXFLAGS += --ptxas-options=-v
endif

CPPFLAGS ?= $(MAKE_CPPFLAGS)
CXXFLAGS ?= $(MAKE_CXXFLAGS)
LDFLAGS ?= -L$(CUDA_PREFIX)/lib64 -lcudart -lcublas -lm -fopenmp 
GPU_CPPFLAGS ?= $(MAKE_GPU_CPPFLAGS)
GPU_CXXFLAGS ?= $(MAKE_GPU_CXXFLAGS)

CPU_COMPILE = $(CXX) $(CPPFLAGS) $(CXXFLAGS)
GPU_COMPILE = $(GPU_CXX) $(CPPFLAGS) $(GPU_CPPFLAGS) $(GPU_CXXFLAGS)

PROGRAMS = dmv_main
dmv_OBJECTS = alloc.o dmv.o dmv_gpu.o dmv_main.o error.o gpu_util.o  timer.o

all: $(PROGRAMS)

dmv_main: $(dmv_OBJECTS)
	$(CXX) -o $@ $(dmv_OBJECTS) $(LDFLAGS)

gpu_util.o: gpu_util.cu
	$(GPU_COMPILE) -o $@ $<

dmv_gpu.o: dmv_gpu.cu
	$(GPU_COMPILE) -o $@ $<

dmv_main.o: dmv_main.cu
	$(GPU_COMPILE) -o $@ $<

%.o: %.c
	$(CPU_COMPILE) -o $@ $<

%.o: %.h
dmv_gpu.o: dmv.h
dmv_main.o: alloc.h dmv.h error.h gpu_util.h timer.h

clean:
	$(RM) -f $(PROGRAMS) $(dmv_OBJECTS)

query:
	@echo '\n' | $(CUDA_PREFIX)/samples/C/bin/linux/release/deviceQuery

help:
	@echo 'Mini help for using this Makefile:'; \
	echo '\tBuild release version: make DEBUG=0 [default: 1]'; \
	echo '\tDisable building of the serial kernel: make SERIAL=0 [default: 1]'; \
	echo '\tDisable building of the OpenMP kernel: make OPENMP=0 [default: 1]'; \
	echo '\tDisable building of the GPU kernels: make GPU=0 [default: 1]'; \
	echo '\tDisable printing of GPU register info: make REGINFO=0 [default: 1]'; \
	echo 'You can also customly set CXX, CPPFLAGS, CXXFLAGS and LDFLAGS as usual. Use the GPU_ prefix for gpu-specific flags.'; \
	echo 'NOTE: Disabling the serial kernel, also disables results checking. Consider avoiding this feature when debugging.'
