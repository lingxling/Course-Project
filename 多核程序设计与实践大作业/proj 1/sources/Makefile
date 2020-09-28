CU_APPS=hw1_cuda
C_APPS=hw1_omp    	# specify the name of OpenMP file (exclude .c) 

GCC_HOME=/home/jovyan/gcc6/bin/gcc    # specify your own gcc install path

all: ${CU_APPS} ${C_APPS}

%: %.cu
	nvcc -o $@ $<
    
%: %.c
	${GCC_HOME} -fopenmp -lm -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}