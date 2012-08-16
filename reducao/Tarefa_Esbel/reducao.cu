#include "cuda.h"
#include <stdio.h>

#define imin(a,b) (a<b?a:b)

// const int N = 33 * 1024;
const int N = 100;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin( 32, (N+threadsPerBlock-1) / threadsPerBlock );
			
__global__ void reduction( float *in, float *out, int n ) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

	cache[cacheIndex] = (tid < n)? in[cacheIndex] : 0;
    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        out[blockIdx.x] = cache[0];
} 

__global__ void add( float *a, float *b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
    if (tid < N)
        a[tid] = a[tid] + b[tid];
}

int main (void){
	
	FILE *aqv = NULL;
	
	aqv = fopen("entrada.txt", "r");
	if(aqv == NULL)
		printf("\nErro ao abrir aqv\n");
	
	double *d = (double *) calloc(N, sizeof(double));
	double *d_old = (double *) calloc(N, sizeof(double));
	float *f = (float *) calloc(N, sizeof(float));
	float *f_old = (float *) calloc(N, sizeof(float));
	
	/**
	 * 
	 * inicia valores
	 * 
	 */
	int i;
	for(i = 0; i < N; i++){
		fscanf (aqv, "%f", &f[i]);
		f_old[i] = 1.0;
	}
	
	/**
	 * 
	 * calcula a soma dos vetores
	 * Efetua a soma da matriz f
	 */
	float soma = 0.0;
	float temp = 0.0;
	for(i = 0; i < N; i++){
		temp = f[i] + f_old[i];
		soma += temp;
	}
	
	/**
	 * 
	 * Pula o ponteiro de leitura do arquivo
	 * devolta para o inicio do aqv
	 * 
	 */
	fseek ( aqv, 0, SEEK_SET);
	
	/**
	 * 
	 * inicia valores para o tipo DOUBLE
	 */
	for(i = 0; i < N; i++){
		fscanf (aqv, "%lf", &d[i]);
		d_old[i] = 1.0;
	}
	
	/**
	 * 
	 * calcula a soma dos vetores
	 * Efetua a soma da matriz f
	 */
	double somad = 0.0;
	double tempd = 0.0;
	
	for(i = 0; i < N; i++){
		tempd = d[i] + d_old[i];
		somad += tempd;
	}
	
	printf("\ncom double SERIAL - Valor da soma = %f\n", somad);
	printf("\ncom double SERIAL - Valor da media = %f\n", somad/N);
	printf("\ncom float SERIAL - Valor da soma = %f\n", soma);
	printf("\ncom float SERIAL - Valor da media = %f\n\n\n", soma/N);
	
	float   *partial_c = NULL;
	float   *dev_f = NULL, *dev_f_old = NULL, *dev_partial_c = NULL;
	
	partial_c = (float*) calloc( blocksPerGrid, sizeof(float) );
	
	cudaMalloc( (void**)&dev_f, N * sizeof(float) );
	cudaMalloc( (void**)&dev_f_old, N * sizeof(float) );
    cudaMalloc( (void**)&dev_partial_c, blocksPerGrid * sizeof(float) );
	
	cudaMemcpy( dev_f, f, N*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_f_old, f_old, N*sizeof(float), cudaMemcpyHostToDevice );
	
	
	add<<<blocksPerGrid,threadsPerBlock>>>( dev_f, dev_f_old );
	
	cudaDeviceSynchronize();
	
	reduction<<<blocksPerGrid,threadsPerBlock>>>( dev_f, dev_partial_c, 100 );
	
	cudaMemcpy( partial_c, dev_partial_c,
                              blocksPerGrid*sizeof(float),
                              cudaMemcpyDeviceToHost );
	soma = 0.0;
	double teste = 0.0;
    for ( i=0; i<blocksPerGrid; i++) {
        soma += partial_c[i];
		teste += partial_c[i];
    }
    
    printf("\n com double CUDA - Valor da soma = %f\n", teste);
	printf("\n com double CUDA - Valor da media = %f\n", teste/N);
    printf("\n CUDA - Valor da soma = %f\n", soma);
	printf("\n CUDA - Valor da media = %f\n", soma/N);
    
    cudaFree( dev_f );
    cudaFree( dev_partial_c );
	
	free(f);
	free(partial_c);
	
	return 0;
}