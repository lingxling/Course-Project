/* ******************************************************** */
/* 文件说明：文件实现了用OpenMP并行计算二维高斯分布。
/*          输入整数s，输出(6s+1)*(6s+1)的二维高斯分布矩阵。
/* ******************************************************** */
#define PI 3.14159265359f
#define THREAD_NUM 100
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

/* ****************************************************************************************************************** */
/* 函数名：GaussianFunction 
/* 函数描述：输入坐标(x,y)和二维高斯分布的标准差sigma，输出点(x,y)对应的二维高斯分布概率密度值。(此处高斯分布的期望值为0向量)
/* 参数描述：
/* x, y: 确定平面上的点(x, y)
/* sigma: 二维高斯分布的标准差
/* ****************************************************************************************************************** */
float GaussianFunction(int x, int y, int sigma);

/* ********************************************************************************** */
/* 函数名：output
/* 函数描述：float数组中含有length*length个元素，该函数功能为输出float数组中的所有元素
/* 参数描述：
/* arr: 一维float数组，含length*length个元素
/* length: 用于计算arr的大小
/* ********************************************************************************** */
void output(const float* arr, int length);

int main(int argc, char const *argv[])
{
	int i, j;
	int tmp_s;
	if (argc == 1) scanf("%d", &tmp_s);
	else tmp_s = atoi(argv[1]);
	const int s = tmp_s;
	int matrix_length = 6 * s + 1;
    int half_length = 3 * s;
    int matrix_size = matrix_length * matrix_length;
	float* arr = NULL;    

    arr = (float*)malloc(sizeof(float)*matrix_size); /*申请内存，若申请失败则结束程序*/
    if (!arr){
        fprintf(stderr, "Memory allocation error!\n");
        abort();
    }

#pragma omp parallel for
	for (i = 0; i < matrix_length; ++i) {
#pragma omp parallel for num_threads(THREAD_NUM)
		for (j = 0; j < matrix_length; ++j) {
			arr[i*matrix_length + j] = GaussianFunction(i - half_length, j - half_length, s);
		}
	}    
#pragma omp barrier

    output(arr, matrix_length);  
    free(arr);
    return 0;
}

float GaussianFunction(int x, int y, int sigma) {
	float argument = -(x*x + y * y) / (float)(2 * sigma*sigma);
	float coefficient = 1 / (sigma*sqrtf(2 * PI));
	return coefficient * expf(argument);
}

void output(const float* arr, int length) {
	int i, j;
	for (i = 0; i < length; ++i) {
		for (j = 0; j < length; ++j) {
			printf("%5.4f ", arr[i*length + j]);
		}
	}
}
