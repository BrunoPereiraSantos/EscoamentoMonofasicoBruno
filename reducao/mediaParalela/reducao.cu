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

int main (void){
	
	FILE *aqv = NULL;
	
	aqv = fopen("entrada.txt", "r");
	if(aqv == NULL)
		printf("\nErro ao abrir aqv\n");
	
	float *f = (float *) malloc(N * sizeof(float));
	float soma = 0.0;
	int i;
	for(i = 0; i < 100; i++){
		fscanf (aqv, "%f", &f[i]);
		soma += f[i];
	}
	printf("\nSERIAL - Valor da soma = %f\n", soma);
	printf("\nSERIAL - Valor da media = %f\n", soma/64);
	
	float   *partial_c;
	float   *dev_f, *dev_partial_c;
	partial_c = (float*)malloc( blocksPerGrid*sizeof(float) );
	
	cudaMalloc( (void**)&dev_f, N * sizeof(float) );
    cudaMalloc( (void**)&dev_partial_c, blocksPerGrid * sizeof(float) );
	
	cudaMemcpy( dev_f, f, N*sizeof(float), cudaMemcpyHostToDevice );
	
	reduction<<<blocksPerGrid,threadsPerBlock>>>( dev_f, dev_partial_c, 100 );
	
	cudaMemcpy( partial_c, dev_partial_c,
                              blocksPerGrid*sizeof(float),
                              cudaMemcpyDeviceToHost );
	soma = 0.0;
    for ( i=0; i<blocksPerGrid; i++) {
        soma += partial_c[i];
    }
    
    printf("\n CUDA - Valor da soma = %f\n", soma);
	printf("\n CUDA - Valor da media = %f\n", soma/64);
    
    cudaFree( dev_f );
    cudaFree( dev_partial_c );
	
	free(f);
	free(partial_c);
	
	return 0;
}