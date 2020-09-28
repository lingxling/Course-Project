/* ************************************************************** */
/* 文件说明：用cuda实现奇偶移向排序
/* ************************************************************** */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAX_BUF 100000000
#define BLOCK_SIZE 512
#define GRID_SIZE 128
#define gpu_error_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }

typedef unsigned int UINT;
UINT buffer[MAX_BUF];

//读文件作为输入数据
UINT ReadFile(const char *szFile, UINT data[])
{
	UINT len;
	FILE *fp;
	fp = fopen(szFile, "rb");
	fread(&len, sizeof(UINT), 1, fp);
	if (len > MAX_BUF)
	{
		fclose(fp);
		return 0;
	}
	fread(data, sizeof(UINT), len, fp);
	fclose(fp);
	return len;
}

//将排好序的数据输出
void WriteFile(const char *szFile, UINT data[], UINT len)
{
	FILE *fp;
	if (len > MAX_BUF)
		return;
	fp = fopen(szFile, "wb");
	fwrite(&len, sizeof(UINT), 1, fp);
	fwrite(data, sizeof(UINT), len, fp);
	fclose(fp);
}

/* ************************************************************************** */
/* 【函数名】：kernel
/* 【函数描述】：用Nvidia显卡并行执行，执行一次代表一个奇数步或者偶数步
/* 【参数描述】：
/* dev_data: 待排序的数据
/* length: 数据长度
/* cur_step: 当前步序号，用于判断当前步是奇数步还是偶数步
/* ************************************************************************** */
__global__ void kernel(UINT* dev_data, int length, int cur_step){
	//当i=0, 2, 4,...时为奇数步，此时有is_even=0; 
	//当i=1, 3, 5,...时为偶数步，此时有is_even=1;
	int is_even = cur_step%2;  //判断当前步是奇数步还是偶数步
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
	while((2*tid+1+is_even) < length){
		//若为奇数步，则对d[0], d[2], d[4],... 排序
		//若为偶数步，则对d[1], d[3], d[5],... 排序
		int d0 = dev_data[is_even + 2*tid];  
		int d1 = dev_data[is_even + 2*tid+1];
		if (d0 > d1){
			dev_data[is_even + 2*tid] = d1;
			dev_data[is_even + 2*tid+1] = d0;
		}
		tid += blockDim.x * gridDim.x;
	}
}

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
  
int main(int argc, char *argv[])
{
	UINT length;
	if (argc != 2)
		return 1;
	length = ReadFile(argv[1], buffer); 

	UINT* dev_data = NULL;
	gpu_error_check(cudaSetDevice(0));  //选择第0块GPU运行
	gpu_error_check(cudaMalloc((void**)&dev_data, sizeof(UINT)*length));  //申请GPU内存
	gpu_error_check(cudaMemcpy(dev_data, buffer, sizeof(UINT)*length, cudaMemcpyHostToDevice));  //将数据从CPU内存拷贝至GPU内存
	for (int i = 0; i < length; ++i){  
		kernel<<<GRID_SIZE, BLOCK_SIZE>>>(dev_data, length, i);  //执行一个奇数步或者偶数步
		gpu_error_check(cudaStreamSynchronize(0));  //同步所有线程
	}
	gpu_error_check(cudaMemcpy(buffer, dev_data, sizeof(UINT)*length, cudaMemcpyDeviceToHost));  //将数据从GPU端拷贝至CPU端
	gpu_error_check(cudaFree(dev_data));  //释放内存
	WriteFile("output.bin", buffer, length);
	return 0;
}
