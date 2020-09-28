/* ************************************************************** */
/* 文件说明：文件实现了用CUDA并行计算二维高斯核，并用高斯核对图像卷积
/*          输入整数s，输出卷积后的数据。
/* ************************************************************** */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAX_BUFFER 1000000
#define MAX_KERNEL_SIZE 128
#define BLOCK_SIZE 64
#define GRID_SIZE 32
#define PI 3.14159265359f
#define gpu_error_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }

int height, width;
float data[MAX_BUFFER];
__device__ __constant__ float constant_gaussian_kernel[MAX_KERNEL_SIZE*MAX_KERNEL_SIZE];

/* ************************************************************************** */
/* 【函数名】：gpu_assert
/* 【函数描述】：用于检测cuda函数是否运行成功。若不成功则输出错误代码和错误文件及行数
/* 【参数描述】：
/* code：指示cuda函数运行出错对应的错误码
/* file：指明出错文件
/* line：指明出错行数
/* abort: 指明出错后是否需要退出程序
/* ************************************************************************** */
inline void gpu_assert(cudaError_t code, const char *file, const int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/* **************************************************************************************** */
/* 【函数名】：gaussian_convolution_with_cuda
/* 【函数描述】：CPU端调用，CPU端执行。该函数会调用CUDA核函数，求出高斯卷积的结果并保存在results中
/* 【参数描述】：
/* results: 用于保存高斯卷积的结果。
/* image_height：被卷积图像的高
/* image_width：被卷积图像的宽
/* gaussian_kernel_size: 高斯核的行（列）。
/* **************************************************************************************** */
void gaussian_convolution_with_cuda(float* results, const int image_height, const int image_width, const int gaussian_kernel_size);


/* **************************************************************************************************************** */
/* 【函数名】：compute_gaussian_distribution
/* 【函数描述】：CUDA 核函数，从CPU调用，在GPU端执行。
/* 用GPU算出高斯核中第(i*length+j)个元素在平面上对应的点(i,j)的二维高斯函数的值，并将计算结果写入gaussian_kernel对应的位置。
/* 【参数描述】：
/* gaussian_kernel: 高斯核，大小为gaussian_kernel_size*gaussian_kernel_size，用于保存结果。
/* gaussian_kernel_size: 高斯核的行（列）。
/* **************************************************************************************************************** */
__global__ void compute_gaussian_distribution(float* gaussian_kernel, const int gaussian_kernel_size);


/* *************************************************************************************************** */
/* 【函数名】：kernel
/* 【函数描述】：
/* CUDA 核函数，从CPU调用，在GPU端执行。
/* 这个函数对应高斯核比较小，可以放在GPU端常量内存中的情况。用GPU算出高斯核和图像的卷积结果，并保存在results中。
/* 【参数描述】：
/* results: GPU动态内存，大小为image_height*image_width，用于保存结果。
/* dev_data: 原图像
/* image_height：被卷积图像的高
/* image_width：被卷积图像的宽
/* gaussian_kernel_size: 高斯核的行（列）。
/* *************************************************************************************************** */
__global__ void kernel(float* results, float* dev_data, const int image_height, const int image_width, const int gaussian_kernel_size);


/* *************************************************************************************************** */
/* 【函数名】：convolution_with_large_gaussian_kernel
/* 【函数描述】：
/* CUDA 核函数，从CPU调用，在GPU端执行。
/* 这个函数对应高斯核比较大，只能放在GPU端动态内存中的情况。用GPU算出高斯核和图像的卷积结果，并保存在results中。
/* 【参数描述】：
/* results: GPU动态内存，大小为image_height*image_width，用于保存结果。
/* dev_data: 原图像
/* image_height：被卷积图像的高
/* image_width：被卷积图像的宽
/* gaussian_kernel_size: 高斯核的行（列）。
/* *************************************************************************************************** */
__global__ void convolution_with_large_gaussian_kernel(float* results, float* dev_data, float* gaussian_kernel, const int image_height, const int image_width, const int gaussian_kernel_size);


int main(int argc, char *argv[]) {
	FILE *fp;
	fp = fopen(argv[1], "rb");
	fread(&height, sizeof(height), 1, fp);
	fread(&width, sizeof(width), 1, fp);
	fread(data, sizeof(float), height*width, fp);
	fclose(fp);
	
	const int gaussian_kernel_size = 6 * atoi(argv[2]) + 1;
	float* convolution_results = NULL;  
	convolution_results = (float*)malloc(sizeof(float)*height*width); /*申请CPU端的动态内存*/
	if (convolution_results==NULL){
		printf("Malloc failed, abort.\n");
		abort();
	}
	gaussian_convolution_with_cuda(convolution_results, height, width, gaussian_kernel_size);

	printf("%d %d\n", height, width);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%.02f ", convolution_results[i*width + j]);
		}
	}
	free(convolution_results);
	return 0;
}

void gaussian_convolution_with_cuda(float* results, const int image_height, const int image_width, const int gaussian_kernel_size) {
	float* dev_gaussian_kernel = NULL;
	float* dev_results = NULL;
	float* dev_data = NULL;

	gpu_error_check(cudaSetDevice(0));  /*选择第0块GPU运行*/
	gpu_error_check(cudaMalloc((void**)&dev_gaussian_kernel, sizeof(float)*gaussian_kernel_size*gaussian_kernel_size));
	gpu_error_check(cudaMalloc((void**)&dev_results, sizeof(float)*image_height*image_width));
	gpu_error_check(cudaMalloc((void**)&dev_data, sizeof(float)*image_height*image_width));
	
	/*创建两个流，使得计算高斯核的过程和将数据拷贝至GPU端的过程可以同时进行*/
	cudaStream_t stream0, stream1;  
	cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
	gpu_error_check(cudaMemcpyAsync(dev_data, data, sizeof(float)*image_height*image_width, cudaMemcpyHostToDevice, stream0));
	compute_gaussian_distribution<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(dev_gaussian_kernel, gaussian_kernel_size);
	gpu_error_check(cudaGetLastError());  /*检测用core并行计算时是否出现错误*/
	gpu_error_check(cudaDeviceSynchronize()); /*同步所有线程*/
	cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1); 

    if (gaussian_kernel_size <= 128){  /*判断高斯核的大小，如果高斯核过大，则不能放在常量内存中，必须用动态内存保存*/
        gpu_error_check(cudaMemcpyToSymbol(constant_gaussian_kernel, dev_gaussian_kernel, sizeof(float)*gaussian_kernel_size*gaussian_kernel_size, 0, cudaMemcpyDeviceToDevice));
        kernel<<<GRID_SIZE, BLOCK_SIZE>>>(dev_results, dev_data, image_height, image_width, gaussian_kernel_size);
        gpu_error_check(cudaDeviceSynchronize()); /*同步所有线程*/
    }

    else{
        convolution_with_large_gaussian_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(dev_results, dev_data, dev_gaussian_kernel, image_height, image_width, gaussian_kernel_size);
        gpu_error_check(cudaGetLastError());
        gpu_error_check(cudaDeviceSynchronize()); /*同步所有线程*/
    }

    gpu_error_check(cudaMemcpy(results, dev_results, sizeof(float)*image_height*image_width, cudaMemcpyDeviceToHost));
	gpu_error_check(cudaFree(dev_gaussian_kernel));
	gpu_error_check(cudaFree(dev_results));
    gpu_error_check(cudaFree(dev_data));
}

__global__ void compute_gaussian_distribution(float* gaussian_kernel, const int gaussian_kernel_size) {
	const int s = (gaussian_kernel_size - 1) / 6;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int n = gaussian_kernel_size * gaussian_kernel_size;
	int i, j, x, y = 0;  /*arr[i][j] = g(x, y)*/
	while (tid < n) {
		i = tid / gaussian_kernel_size;
		j = tid - i * gaussian_kernel_size;
		x = i - 3 * s;
		y = j - 3 * s;
		gaussian_kernel[tid] = 1 / (s*sqrtf(2 * PI)) * expf(-(x*x + y * y) / float(2 * s*s));
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void kernel(float* results, float* dev_data, const int image_height, const int image_width, const int gaussian_kernel_size) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int n = image_height * image_width;
	int s = (gaussian_kernel_size - 1) / 6;
	int x, y, i, j = 0;  /*x, y:gaussian kernel index; i, j: image index*/
	int i1, j1 = 0;  /*i1 = i-3s, y1 = j-3s*/
	float tmp = 0;
	while (tid < n) {
		i = tid / image_width;
		j = tid - i * image_width;
		tmp = 0;
		for (x = 0; x < gaussian_kernel_size; ++x) {
			for (y = 0; y < gaussian_kernel_size; ++y) {
				i1 = i + x - 3 * s;
				j1 = j + y - 3 * s;
				if (i1>=0 && i1<image_height && j1>=0 && j1<image_width)
					tmp += dev_data[i1 * image_width + j1] * constant_gaussian_kernel[x*gaussian_kernel_size + y];
			}
		}
		results[tid] = tmp;
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void convolution_with_large_gaussian_kernel(float* results, float* dev_data, float* gaussian_kernel, const int image_height, const int image_width, const int gaussian_kernel_size){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int n = image_height * image_width;
	int s = (gaussian_kernel_size - 1) / 6;
	int x, y, i, j = 0;  /*x, y:gaussian kernel index; i, j: image index*/
	int i1, j1 = 0;  /*i1 = i-3s, y1 = j-3s*/
	float tmp = 0;
	while (tid < n) {
		i = tid / image_width;
		j = tid - i * image_width;
		tmp = 0;
		for (x = 0; x < gaussian_kernel_size; ++x) {
			for (y = 0; y < gaussian_kernel_size; ++y) {
				i1 = i + x - 3 * s;
				j1 = j + y - 3 * s;
				if (i1>=0 && i1<image_height && j1>=0 && j1<image_width)
					tmp += dev_data[i1 * image_width + j1] * gaussian_kernel[x*gaussian_kernel_size + y];
			}
		}
		results[tid] = tmp;
		tid += blockDim.x * gridDim.x;
	}
}
