/* ******************************************************** */
/* 文件说明：文件实现了用CUDA并行计算二维高斯分布。
/*          输入整数s，输出(6s+1)*(6s+1)的二维高斯分布矩阵。
/* ******************************************************** */
#define PI 3.14159265359f
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32
#define GRID_M 16
#define GRID_N 7
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#include <cstdio>
#include <cmath>
#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* ************************************************************************** */
/* 函数名：gpuAssert
/* 函数描述：用于检测cuda函数是否运行成功。若不成功则输出错误代码和错误文件及行数
/* 参数描述：
/* code：指示cuda函数运行出错对应的错误码
/* file：指明出错文件
/* line：指明出错行数
/* abort: 指明出错后是否需要退出程序
/* ************************************************************************** */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/* ***************************************************************************** */
/* 函数名：calculateGaussianWithCuda
/* 函数描述：
/* 从CPU端调用，在CPU端执行。给定大小为length*length的一维数组arr(在CPU端的内存中)。
/* 该函数申请GPU端内存，【调用】相关CUDA函数，并把高斯函数计算结果存入arr。
/* 参数描述：
/* arr: 大小为length*length的一维数组，数组由CPU端的内存动态分配
/* length: 用于【计算】arr的界限
/* ***************************************************************************** */
cudaError_t calculateGaussianWithCuda(float* arr, const int length);

/* *************************************************************************************** */
/* 函数名：gaussianDistribution
/* 函数描述：
/* CUDA 核函数，从CPU调用，在GPU端执行。
/* 用GPU算出数组中第(j*length+i)个元素在平面上对应的点(i,j)的二维高斯函数的值并写入arr对应的位置，
/* arr的实际大小是大于等于length*length的，一般为128byte的倍数，目的是为了访问内存时可以对齐访问。
/* 因此，实际上用arr[j*pitch+i]来存点(i,j)的二维高斯函数。
/* 参数描述：
/* arr: 大小为length*length的一维数组，数组由GPU的内存动态分配
/* matrix_length: 用于【计算】arr的界限
/* pitch：每次访问内存可以取得的连续的float数组的大小
/* *************************************************************************************** */
__global__ void gaussianDistribution(float* arr, const int matrix_length, int pitch);

/************************************************************************ */
/* 函数名：output
/* 函数描述：
/* float数组中含有length*length个元素，该函数功能为输出float数组中的所有元素
/* 参数描述：
/* arr: 一维float数组，含length*length个元素
/* length: 用于计算arr的大小
/* *********************************************************************** */
__host__ void output(float* arr, const int length);


int main(int argc, char const *argv[]) {
	int tmp_s;
	if (argc==1) scanf("%d", &tmp_s);
	else tmp_s = atoi(argv[1]);
	const int s = tmp_s;
	const int matrix_length = 6 * s + 1;
	float* arr = NULL;

	try {
		arr = new float[matrix_length*matrix_length]; /*申请CPU端的动态内存*/
	}
	catch (const std::bad_alloc& e) {
		printf(e.what());
		abort();
	}
	calculateGaussianWithCuda(arr, matrix_length);  /*调用GPU并行计算*/
	gpuErrchk(cudaDeviceReset());
	output(arr, matrix_length);
	delete[] arr;
	return 0;
}

cudaError_t calculateGaussianWithCuda(float* arr, const int matrix_length) {
	float* dev_arr = NULL;
	size_t pitch;
	cudaError_t cuda_status = cudaSuccess;
	
	gpuErrchk(cudaSetDevice(0));  /*选择第0块GPU运行*/
	gpuErrchk(cudaMallocPitch(&dev_arr, &pitch, matrix_length * sizeof(float), matrix_length));  /*申请GPU内存*/
	
	dim3 block_dim(BLOCK_WIDTH, BLOCK_HEIGHT);
	dim3 grid_dim(GRID_M, GRID_N);
	gaussianDistribution <<<grid_dim, block_dim >>> (dev_arr, matrix_length, pitch / sizeof(float));
	
	gpuErrchk(cudaGetLastError());  /*检测用core并行计算时是否出现错误*/
	gpuErrchk(cudaDeviceSynchronize()); /*同步所有线程*/

	/*将GPU内存中的结果赋值到CPU内存*/
	gpuErrchk(cudaMemcpy2D(arr, sizeof(float)*matrix_length, dev_arr, pitch, sizeof(float)*matrix_length, matrix_length, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(dev_arr));
	return cuda_status;
}

__global__ void gaussianDistribution(float* dev_arr, const int matrix_length, int pitch) {
	int s = (matrix_length - 1) / 6;

	for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < matrix_length; y += blockDim.y * gridDim.y)
	{
		for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < matrix_length; x += blockDim.x * gridDim.x)
		{
			int i = x - 3 * s;
			int j = y - 3 * s;
			float tmp = 1 / (s*sqrtf(2 * PI)) * expf(-(i*i + j * j) / float(2 * s*s));
			dev_arr[y*pitch + x] = tmp;
		}
	}
}

__host__ void output(float* arr, const int length) {
	int matrix_size = length * length;
	for (int i = 0; i < matrix_size; ++i) {
		printf("%5.4f ", arr[i]);
	}
}
