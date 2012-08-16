#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define N       100
#define DIM      2
#define PamM 2e-11
#define S 0.5

char le_entrada();
char inicializa_parametros();
float *aloca_matriz(int, int);
void cal_cond_robin();
char parametro_independentes();
char copia_dados_para_gpu();
void copia_dados_para_cpu();
void clear_mem();
//char calcula_pressao_velocidade(int, int, int, int, int);
//char atualiza_mult_lagrange(int tid);

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

//- - - - - - - - - - - - - - GLOBAIS - - - - - - - - - - - - - - //



/* - - - - - - - Estruturas  - - - - - - - */

typedef struct{
	float *R, *L, *U, *D;
	float *R_old, *L_old, *U_old, *D_old;
}ESTRUTURA_Q;

typedef struct{
	float *R, *L, *U, *D;
	float *R_old, *L_old, *U_old, *D_old;
}ESTRUTURA_L;

typedef struct{
	float *R, *L, *U, *D;
	float *R_old, *L_old, *U_old, *D_old;
}ESTRUTURA_B;

typedef struct{
	float *p, *p_old;
}ESTRUTURA_PRESSAO;

typedef struct{
	float *perm, *font, *epsilon;
}ESTRUTURA_MAT;

/* - - - - - - - Fim das Estruturas  - - - - - - - */

/* - - - - - - - Variaveis das Estruturas  - - - - - - - */
ESTRUTURA_Q host_q, dev_q;
ESTRUTURA_L host_l, dev_l;
ESTRUTURA_B host_b, dev_b;
ESTRUTURA_PRESSAO host_pressao, dev_pressao;
ESTRUTURA_MAT host_mat, dev_mat;

/* - - - - - - - Entradas Externas  - - - - - - - */
int tam_mat_interna = 3, tam_mat_real = 3 + 2, max_interacoes = 1000, op_contorno = 1;
float tam_regiao = 20000.00, erro_max = 1e-5, valor_contor = 2.00;
float h = 20000.00 / 3; // ALTURA H = TAM_REGIAO / TAM_MAT_INTERNA
//float *mat_perm = NULL, *mat_font = NULL, *mat_epsilon = NULL;
//float *dev_mat_perm = NULL, *mat_font = NULL, *mat_epsilon = NULL;
/* - - - - - - - Fim das Entradas Externas  - - - - - - - */
/* - - - - - - - Fim das Variaveis das Estruturas  - - - - - - - */

/* - - - - - - - Ponteiros para GPU  - - - - - - - */
float *dev_aux = NULL, dev_erro = NULL, *dev_media =  NULL;
// float *dev_aux = NULL, dev_erro = 0.0, dev_media = 0.0, dev_sum1 = 0.0, dev_sum2 = 0.0;
// 
// float *dev_q.R = NULL, *dev_q.L = NULL, *dev_q.U = NULL, *dev_q.D = NULL;
// float *dev_q.R_old = NULL, *dev_q.L_old = NULL, *dev_q.U_old = NULL, *dev_q.D_old = NULL;
// 
// float *dev_l.R = NULL, *dev_l.L = NULL, *dev_l.U = NULL, *dev_l.D = NULL;
// float *dev_l.R_old = NULL, *dev_l.L_old = NULL, *dev_l.U_old = NULL, *dev_l.D_old = NULL;
// 
// float *dev_b.R = NULL, *dev_b.L = NULL, *dev_b.U = NULL, *dev_b.D = NULL;
// float *dev_b.R_old = NULL, *dev_b.L_old = NULL, *dev_b.U_old = NULL, *dev_b.D_old = NULL;
// 
// float *dev_pressao.p = NULL, *dev_pressao.p_old = NULL;
// 


//- - - - - - - - - - - - - - FIM - GLOBAIS - - - - - - - - - - - - - - //
__device__ char atualiza_mult_lagrange(	int tid,
										ESTRUTURA_Q *dev_q,
										ESTRUTURA_L *dev_l,
										ESTRUTURA_B *dev_b
										){
	
	int index_mem_central = 0, index_mem_down = 0, index_mem_uper = 0;
	int index_mem_left = 0, index_mem_right = 0;
	int offset = (blockDim.x * gridDim.x); // o kernel contem somente a quantidade de elementos internos
												   // portanto a fronteira deve ser contata "+ 2" de cada lado
	
	index_mem_central = tid;
	index_mem_uper = index_mem_central - offset; // (offset -1) = comprimento do kernel
	index_mem_down = index_mem_central + offset;
	index_mem_left = index_mem_central - 1;
	index_mem_right = index_mem_central + 1;
	
	dev_l->U[index_mem_central] = dev_b->U[index_mem_central] * (dev_q->U[index_mem_central] + dev_q->D_old[index_mem_uper]) + dev_l->D_old[index_mem_uper];
	
	dev_l->D[index_mem_central] = dev_b->D[index_mem_central] * (dev_q->D[index_mem_central] + dev_q->U_old[index_mem_down]) + dev_l->U_old[index_mem_down];
	
	dev_l->R[index_mem_central] = dev_b->R[index_mem_central] * (dev_q->R[index_mem_central] + dev_q->L_old[index_mem_right]) + dev_l->L_old[index_mem_right];
	
	dev_l->L[index_mem_central] = dev_b->L[index_mem_central] * (dev_q->L[index_mem_central] + dev_q->R_old[index_mem_left]) + dev_l->R_old[index_mem_left];
	
	return 0;
}

__device__ char calcula_pressao_velocidade(	int tid, int uper, int right, int down, int left,
											ESTRUTURA_Q *dev_q,
											ESTRUTURA_L *dev_l,
											ESTRUTURA_B *dev_b,
											ESTRUTURA_PRESSAO *dev_pressao,
											ESTRUTURA_MAT *dev_mat
											){
	
	float auxU = 0.0, auxD = 0.0, auxR = 0.0, auxL = 0.0, DU = 0.0, DD = 0.0, DR = 0.0, DL = 0.0;
	int index_mem_central = 0, index_mem_down = 0, index_mem_uper = 0;
	int index_mem_left = 0, index_mem_right = 0;
	int offset = (blockDim.x * gridDim.x); // o kernel contem somente a quantidade de elementos internos
												   // portanto a fronteira deve ser contata "+ 2" de cada lado
	
	index_mem_central = tid;
	index_mem_uper = index_mem_central - offset;
	index_mem_down = index_mem_central + offset;
	index_mem_left = index_mem_central - 1;
	index_mem_right = index_mem_central + 1;
	
	if(uper == 1){
		auxU = dev_mat->epsilon[index_mem_central] / (1 + dev_b->U[index_mem_central] * dev_mat->epsilon[index_mem_central]);
		DU = auxU * (dev_b->U[index_mem_central] * dev_q->D_old[index_mem_uper] + dev_l->D_old[index_mem_uper]);
	}
	
	if(right == 1){
		auxR = dev_mat->epsilon[index_mem_central] / (1 + dev_b->R[index_mem_central] * dev_mat->epsilon[index_mem_central]);
		DR = auxR * (dev_b->R[index_mem_central] * dev_q->L_old[index_mem_right] + dev_l->L_old[index_mem_right]);
	}
	
	if(down == 1){
		auxD = dev_mat->epsilon[index_mem_central] / (1 + dev_b->D[index_mem_central] * dev_mat->epsilon[index_mem_central]);
		DD = auxD * (dev_b->D[index_mem_central] * dev_q->U_old[index_mem_down] + dev_l->U_old[index_mem_down]);
	}
	
	if(left == 1){
		auxL = dev_mat->epsilon[index_mem_central] / (1 + dev_b->L[index_mem_central] * dev_mat->epsilon[index_mem_central]);
		DL = auxL * (dev_b->L[index_mem_central] * dev_q->R_old[index_mem_left] + dev_l->R_old[index_mem_left]);
	}
	
	dev_pressao->p[index_mem_central] = (dev_mat->font[index_mem_central] + DU + DR + DD + DL) / (auxU + auxR + auxD + auxL);
	
	dev_q->L[index_mem_central] = auxL * dev_pressao->p[index_mem_central] - DL;
	dev_q->R[index_mem_central] = auxR * dev_pressao->p[index_mem_central] - DR;
	dev_q->U[index_mem_central] = auxU * dev_pressao->p[index_mem_central] - DU;
	dev_q->D[index_mem_central] = auxD * dev_pressao->p[index_mem_central] - DD;
	
	return 0;
}

__global__ void escoamento_monofasico(	ESTRUTURA_Q dev_q,
										ESTRUTURA_L dev_l,
										ESTRUTURA_B dev_b,
										ESTRUTURA_PRESSAO dev_pressao,
										ESTRUTURA_MAT dev_mat,
										float *dev_aux, const float erro_max, float dev_erro, float *dev_media){
    /*int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

        a[offset] = offset;*/
	/*vificar as condições de contorno*/	
		
		
	float dev_sum1 = 0.0, dev_sum2 = 0.0;
	int flag_thread_centrais = 1;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	/*int offset = (blockDim.x * gridDim.x) + 1; // deslocamento para o tamanho da região (tam_regiao = n + 2)
	*/
	int tid = x + y * blockDim.x * gridDim.x;
	//verificar esse deslocamento para n causar problema (somente na hora de armazenar utilizar o deslocamento)
	//int tid = (x + y * blockDim.x * gridDim.x) + offset; // tid fornece o indice do vetor
	
	int dimensao_x = blockDim.x * gridDim.x;
	int dimensao_y = blockDim.y * gridDim.y;
	int eq_tid_cant_sup_esq = dimensao_x + 1;
	int eq_tid_cant_sup_dir = dimensao_x + (dimensao_x - 2); // posição extremo sup direito
	int eq_tid_cant_inf_dir = (dimensao_x * dimensao_y) - (dimensao_x + 2); // posição extremo inf direito
	int eq_tid_cant_inf_esq = ((dimensao_x) * (dimensao_y - 2)) + 1; // posição extremo inf esquerdo
	
	int offset = (blockDim.x * gridDim.x) + 1 + 2; // o kernel contem somente a quantidade de elementos internos
												   // portanto a fronteira deve ser contata "+ 2" de cada lado
	
	int index_mem_central = tid;
	int i = 0;
	while(i < 1){
		if(tid == eq_tid_cant_sup_esq){//canto superior esquerdo
			/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
			/*
			*	calcula_pressao_velocidade();
			* 
			*	Param: 	ESTRUTURA_Q dev_q,
						ESTRUTURA_L dev_l,
						ESTRUTURA_B dev_b,
						ESTRUTURA_PRESSAO dev_pressao,
						ESTRUTURA_MAT dev_mat
			*
			*/
				calcula_pressao_velocidade(	tid, 0, 1, 1, 0,
											&dev_q,
											&dev_l,
											&dev_b,
											&dev_pressao,
											&dev_mat);
				
				/*
				* 
				* 	atualiza_mult_lagrange();
				* 
				* 	param:	int tid,
							ESTRUTURA_Q dev_q,
							ESTRUTURA_L dev_l,
							ESTRUTURA_B dev_b
				* 
				*/
				atualiza_mult_lagrange(tid, &dev_q, &dev_l, &dev_b);
				
			flag_thread_centrais = 0;
		}
		
		if(tid == eq_tid_cant_sup_dir){//canto superior direito
			/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
				calcula_pressao_velocidade(	tid, 0, 0, 1, 1,
											&dev_q,
											&dev_l,
											&dev_b,
											&dev_pressao,
											&dev_mat);
				
				atualiza_mult_lagrange(tid, &dev_q, &dev_l, &dev_b);
				
			flag_thread_centrais = 0;
		}
		
		if(tid == eq_tid_cant_inf_esq){//canto inferior esquerdo
			/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/

				calcula_pressao_velocidade(	tid, 1, 1, 0, 0,
											&dev_q,
											&dev_l,
											&dev_b,
											&dev_pressao,
											&dev_mat);
											
											
				atualiza_mult_lagrange(tid, &dev_q, &dev_l, &dev_b);
				
			flag_thread_centrais = 0;
		}
		
		if(tid == eq_tid_cant_inf_dir){//canto inferior direito
			/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
				calcula_pressao_velocidade(	tid, 1, 0, 0, 1,
											&dev_q,
											&dev_l,
											&dev_b,
											&dev_pressao,
											&dev_mat);
				
				atualiza_mult_lagrange(	tid, &dev_q, &dev_l, &dev_b);
				
			flag_thread_centrais = 0;
		}
		
		if((tid > eq_tid_cant_sup_esq) && (tid < eq_tid_cant_sup_dir)){//fronteira superior
			/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
				calcula_pressao_velocidade(	tid, 0, 1, 1, 1,
											&dev_q,
											&dev_l,
											&dev_b,
											&dev_pressao,
											&dev_mat);
				
				atualiza_mult_lagrange(tid, &dev_q, &dev_l, &dev_b);
				
			flag_thread_centrais = 0;
		}
		
		if((tid > eq_tid_cant_sup_dir) && (tid < eq_tid_cant_inf_dir) && (tid % dimensao_x == dimensao_x - 2)){ //fronteira direita
			/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
				calcula_pressao_velocidade(	tid, 1, 0, 1, 1,
											&dev_q,
											&dev_l,
											&dev_b,
											&dev_pressao,
											&dev_mat);
				
				atualiza_mult_lagrange(tid, &dev_q, &dev_l, &dev_b);
				
			flag_thread_centrais = 0;
		}
		
		if((tid > eq_tid_cant_inf_esq) && (tid < eq_tid_cant_inf_dir)){ //fronteira inferior
			/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
				calcula_pressao_velocidade(	tid, 1, 1, 0, 1,
											&dev_q,
											&dev_l,
											&dev_b,
											&dev_pressao,
											&dev_mat);
				
				atualiza_mult_lagrange(tid, &dev_q, &dev_l, &dev_b);
				
			flag_thread_centrais = 0;
		}
		
		if((tid > eq_tid_cant_sup_esq) && (tid < eq_tid_cant_inf_dir) && (tid < eq_tid_cant_inf_esq) && (tid % dimensao_x == 1)){//fronteira esquerda
			/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
				calcula_pressao_velocidade(	tid, 1, 1, 1, 0,
											&dev_q,
											&dev_l,
											&dev_b,
											&dev_pressao,
											&dev_mat);
				
				atualiza_mult_lagrange(tid, &dev_q, &dev_l, &dev_b);
				
			flag_thread_centrais = 0;
		}
		
		if(flag_thread_centrais && (tid % dimensao_x >= 2) && (tid % dimensao_x <= (dimensao_x - 3)) &&
			(tid > eq_tid_cant_sup_dir) &&
			(tid < eq_tid_cant_inf_esq)
		){
			/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
				calcula_pressao_velocidade(	tid, 1, 1, 1, 1,
											&dev_q,
											&dev_l,
											&dev_b,
											&dev_pressao,
											&dev_mat);
				
				atualiza_mult_lagrange(tid, &dev_q, &dev_l, &dev_b);
		}
		
		/*
		*
		*SINCRONIZA
		*COMENTARIOS 
			*ALOCAR VARIÁVEL aux com o tamanho de "tids"
			*VERIFICAR ATOMICIDADE PRA VALORES FLOAT
			*VERIFICAR ALOCAÇÃO DAS MEMÓRIAS GLOBAIS
			*alocar memória erro
			*alocar float media = 0.0, sum1 = 0.0, sum2 = 0.0;
		*/
		__syncthreads();
		dev_media[index_mem_central] = dev_pressao.p[index_mem_central];
		int i = dimensao_x * dimensao_y / 2;
		while(){
		}
		
// 		if((tid == eq_tid_cant_inf_dir)){
// 			dev_media[0] = 0.0;
// 			int i = 0, desloc = 0;
// 			for(i = 1; i <= eq_tid_cant_inf_dir; i++){
// 				if(i % (blockDim.x * gridDim.x) == 0)
// 					desloc += 2;
// 				dev_media[0] += dev_pressao.p[(offset) + i + desloc];
// 			}
// 			
// 			dev_media[0] = dev_media[0] / (eq_tid_cant_inf_dir + 1);
// 		}
		__syncthreads();
		
		dev_pressao.p[index_mem_central] -= dev_media[0];
		dev_l.D[index_mem_central] -= dev_media[0];
		dev_l.U[index_mem_central] -= dev_media[0];
		dev_l.L[index_mem_central] -= dev_media[0];
		dev_l.R[index_mem_central] -= dev_media[0];
		
		//avaliando criterio de convergencia
		dev_aux[index_mem_central] = dev_pressao.p[index_mem_central] - dev_pressao.p_old[index_mem_central];
		__syncthreads();
		if(tid == eq_tid_cant_inf_dir){
			dev_erro = 0.0, dev_sum1 = 0.0, dev_sum2 = 0.0;;
			int i = 0, desloc = 0;
			for(i = 0; i <= eq_tid_cant_inf_dir; i++){
				if(i % (blockDim.x * gridDim.x) == 0)
					desloc += 2;
				dev_sum1 += dev_aux[(offset + 1) + i + desloc] * dev_aux[(offset + 1) + i + desloc];
				dev_sum2 += dev_pressao.p[(offset + 1) + i + desloc] * dev_pressao.p[(offset + 1) + i + desloc];
			}
			dev_erro = sqrt(dev_sum1/dev_sum2);
		}
		__syncthreads();
		
		
		if (dev_erro > erro_max){
			return;
		}
		
		dev_pressao.p_old[index_mem_central] = dev_pressao.p[index_mem_central];
		dev_q.U_old[index_mem_central] = dev_q.U[index_mem_central];
		dev_q.R_old[index_mem_central] = dev_q.R[index_mem_central];
		dev_q.L_old[index_mem_central] = dev_q.L[index_mem_central];
		dev_q.D_old[index_mem_central] = dev_q.D[index_mem_central];
		
		dev_l.D_old[index_mem_central] = dev_l.D[index_mem_central];
		dev_l.U_old[index_mem_central] = dev_l.U[index_mem_central];
		dev_l.L_old[index_mem_central] = dev_l.L[index_mem_central];
		dev_l.R_old[index_mem_central] = dev_l.R[index_mem_central];
		
		i++;
	}
	/* 
	 * Imponiendo a media cero na distribuicao de presiones
	 * Calculo de la media
	 */
	/*
	atomicAdd( &media, dev_pressao.p[tid] );
	//atomicSub( &aux[tid], dev_pressao.p[tid] - dev_pressao.p_old[tid] );
	__syncthreads();
	
	dev_pressao.p[tid] -= M;
	dev_l.D[tid] -= M;
	dev_l.U[tid] -= M;
	dev_l.L[tid] -= M;
	dev_l.R[tid] -= M;
	
	//avaliando criterio de convergencia
	aux[tid] = dev_pressao.p[tid] - dev_b.D_old[tid];
	__syncthreads();
	
	atomicAdd( &sum1, aux[tid] * aux[tid] );
	atomicAdd( &sum2, dev_pressao.p[tid] * dev_pressao.p[tid] );
	__syncthreads();
	if(tid == 0)
		erro = sqrt(sum1/sum2);
		
	if (erro < 1e-5) return 0;
	
	p_old[j][k] = p[j][k];
	dev_pressao.p_old[tid] = dev_pressao.p_old[tid];
	dev_q.U_old[tid] = dev_q.U[tid];
	dev_q.R_old[tid] = dev_q.R[tid];
	dev_q.L_old[tid] = dev_q.L[tid];
	dev_q.D_old[tid] = dev_q.D[tid];
	
	dev_l.D_old[tid] = dev_l.D[tid];
	dev_l.U_old[tid] = dev_l.U[tid];
	dev_l.L_old[tid] = dev_l.L[tid];
	dev_l.R_old[tid] = dev_l.R[tid];*/
	
}


int main(void){
	le_entrada();
	inicializa_parametros();
	cal_cond_robin();
 	parametro_independentes();

	copia_dados_para_gpu();
// 	dim3 block(comprimento/16 , altura/16);
// 	dim3 thread(16, 16);
	dim3 block(2, 2);
	dim3 thread(5, 5);
	
	/*
		 *	escoamento_monofasico();
		 * 
		 *	Param: 	ESTRUTURA_Q dev_q,
					ESTRUTURA_L dev_l,
					ESTRUTURA_B dev_b,
					ESTRUTURA_PRESSAO dev_pressao,
					ESTRUTURA_MAT dev_mat,
					float *dev_aux, const float erro_max
		 *
		 */
	
  	escoamento_monofasico<<<block, thread>>>(	dev_q, dev_l, dev_b, dev_pressao, dev_mat,
												dev_aux, 1e-5, dev_erro, dev_media);

	copia_dados_para_cpu();
	
	
	int i = 0, j = 0;
	
	printf("\ntam_mat_interna = %d\n", tam_mat_interna);
	printf("tam_mat_real = %d\n", tam_mat_real);
	printf("max_interacoes = %d\n", max_interacoes);
	printf("op_contorno = %d\n", op_contorno);
	printf("tam_regiao = %f\n", tam_regiao);
	printf("erro_max = %f\n", erro_max);
	printf("valor_contor = %f\n", valor_contor);
	
	printf("\n\n\t\t\tmat_font:\n");
	
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_mat.font[i*tam_mat_real + j]);
		printf("\n");
	}
	
	
	printf("\n\n\t\t\tmat_perm:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_mat.perm[i*tam_mat_real + j]);
			//printf("%12.4E ", host_mat.perm[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\n\t\t\tmat_epsilon:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_mat.epsilon[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n------------------------------------------------------------------------------------------------------------------------------------------\n");
	printf("\n\n\t\t\tbeta U:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_b.U[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\n\t\t\tbeta R:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_b.R[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\n\t\t\tbeta L:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_b.L[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\n\t\t\tbeta D:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_b.D[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n------------------------------------------------------------------------------------------------------------------------------------------\n");
	
	printf("\n\n\t\t\t\tq_U:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_q.U[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\n\t\t\t\tq_R:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_q.R[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\n\t\t\t\tq_L:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_q.L[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\n\t\t\t\tq_D:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_q.D[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\n\t\t\t\tl_U:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_l.U[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\n\t\t\t\tl_R:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_l.R[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\n\t\t\t\tl_L:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_l.L[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\n\t\t\t\tl_D:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_l.D[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\npressao:\n");
	printf("\n\n\t\t\t\tpressao:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_pressao.p[i*tam_mat_real + j]);
		printf("\n");
	}
	/*printf("\n\n\t\t\t\tb_U:\t\t\t\t\t\t\t\t\tb_U_old:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_b.U[i*tam_mat_real + j]);
		printf("| ");
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_b.U_old[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\n\t\t\t\tb_R:\t\t\t\t\t\t\t\t\tb_R_old:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_b.R[i*tam_mat_real + j]);
		printf("| ");
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_b.R_old[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\n\t\t\t\tb_D:\t\t\t\t\t\t\t\t\tb_D_old:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_b.D[i*tam_mat_real + j]);
		printf("| ");
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_b.D_old[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\n\t\t\t\tb_D:\t\t\t\t\t\t\t\t\tb_D_old:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_b.D[i*tam_mat_real + j]);
		printf("| ");
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_b.D_old[i*tam_mat_real + j]);
		printf("\n");
	}
	printf("\n------------------------------------------------------------------------------------------------------------------------------------------\n");
	printf("\npressao:\n");
	printf("\n\n\t\t\t\tpressao:\t\t\t\t\t\t\t\t\tpressao_old:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_pressao.p[i*tam_mat_real + j]);
		printf("| ");
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_pressao.p_old[i*tam_mat_real + j]);
		printf("\n");
	}
	printf("\n------------------------------------------------------------------------------------------------------------------------------------------\n");
	printf("\n\n\t\t\t\tl_U:\t\t\t\t\t\t\t\t\tl_U_old:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_l.U[i*tam_mat_real + j]);
		printf("| ");
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_l.U_old[i*tam_mat_real + j]);
		printf("\n");
	}
	printf("\n\n\t\t\t\tl_R:\t\t\t\t\t\t\t\t\tl_R_old:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_l.R[i*tam_mat_real + j]);
		printf("| ");
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_l.R_old[i*tam_mat_real + j]);
		printf("\n");
	}
	printf("\n\n\t\t\t\tl_D:\t\t\t\t\t\t\t\t\tl_D_old:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_l.D[i*tam_mat_real + j]);
		printf("| ");
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_l.D_old[i*tam_mat_real + j]);
		printf("\n");
	}
	printf("\n\n\t\t\t\tl_L:\t\t\t\t\t\t\t\t\tl_L_old:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_l.L[i*tam_mat_real + j]);
		printf("| ");
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_l.L_old[i*tam_mat_real + j]);
		printf("\n");
	}
	printf("\n------------------------------------------------------------------------------------------------------------------------------------------\n");
	printf("\n\n\t\t\t\tq_U:\t\t\t\t\t\t\t\t\tq_U_old:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_q.U[i*tam_mat_real + j]);
		printf("| ");
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_q.U_old[i*tam_mat_real + j]);
		printf("\n");
	}
	printf("\n\n\t\t\t\tq_R:\t\t\t\t\t\t\t\t\tq_R_old:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_q.R[i*tam_mat_real + j]);
		printf("| ");
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_q.R_old[i*tam_mat_real + j]);
		printf("\n");
	}
	printf("\n\n\t\t\t\tq_D:\t\t\t\t\t\t\t\t\tq_D_old:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_q.D[i*tam_mat_real + j]);
		printf("| ");
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_q.D_old[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\n\t\t\t\tq_L:\t\t\t\t\t\t\t\t\tq_L_old:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_q.L[i*tam_mat_real + j]);
		printf("| ");
		for(j = 0; j < tam_mat_real; j++)
			printf("%12.4E ", host_q.L_old[i*tam_mat_real + j]);
		printf("\n");
	}*/
	
	clear_mem();
// 	
// 	system("pause");
	return 0;
}
 
 char le_entrada(){
	printf("\n\n\t\t - - CARREGANDO ENTRADA - - \n\n");
	FILE *arq = NULL;
	
	//arq = fopen("../dir_entrada/parametro_entrada.txt", "r");
	arq = fopen("parametro_entrada.txt", "r");
	if(arq == NULL){
		printf("Erro ao abrir aquivo: 'parametro_entrada.txt'\n\t\tCertifique-se que o arquivo exite.\n");
		exit(1);
	}
	else{
		printf("\t\t - - LENDO ARQUIVO DE ENTRADA - -\n");
		/*char c[2], dados[255], buffer[255];*/
		char buffer[255];
		int cont = 1;
		while(cont < 9){
			fscanf(arq, "%s", buffer);
			//puts(buffer);
			int i = 0, j = 0;
			switch(strlen(buffer)){
				case 8: //erro_maximo
				
					fscanf(arq, "%f", &erro_max);
					break;
					
				case 10: //tam_regiao
				
					fscanf(arq, "%f", &tam_regiao);
					break;
					
				case 11: //opcao_contorno
				
					fscanf(arq, "%d", &op_contorno);
					break;
					
				case 12: //valor_contor
				
					fscanf(arq, "%f", &valor_contor);
					break;
					
				case 14: //max_interacoes
				
					fscanf(arq, "%d", &max_interacoes);
					break;
					
				case 15: //tam_mat_interna
				
					fscanf(arq, "%d", &tam_mat_interna);
					break;
					
				case 16: //matriz_de_fontes
					//uso (tam_mat_interna + 2) - pois ainda não inicializei 'tam_mat_real'
					host_mat.font = aloca_matriz(tam_mat_interna + 2, tam_mat_interna + 2);
					for(i = 1; i < (tam_mat_interna + 2) - 1; i ++)
						for(j = 1; j < (tam_mat_interna + 2) - 1 ; j++)
							fscanf(arq, "%f", &host_mat.font[i*(tam_mat_interna+2) + j]);
							
					break;
					
				case 18: //matriz_permeabilidade
				
					host_mat.perm = aloca_matriz(tam_mat_interna + 2, tam_mat_interna + 2);
					host_mat.epsilon = aloca_matriz(tam_mat_interna + 2, tam_mat_interna + 2);
					for(i = 1; i < (tam_mat_interna + 2) - 1; i ++)
						for(j = 1; j < (tam_mat_interna + 2) - 1 ; j++)
							fscanf(arq, "%f", &host_mat.perm[i*(tam_mat_interna+2) + j]);
					
					for(i = 1; i < (tam_mat_interna + 2) - 1; i ++)
						for(j = 1; j < (tam_mat_interna + 2) - 1 ; j++)
							host_mat.perm[i*(tam_mat_interna+2) + j] = PamM*exp(S * host_mat.perm[i*(tam_mat_interna+2) + j]);
					break;
					
				default:
					
					printf("\n\n\t\tHouve algum erro no aquivo de entrada!\n\n");
					return 0;
			}
			//int tam = strlen(buffer);
			cont++;
		}
		printf("\t\t - - ARQUIVO DE ENTRADA CARREGADO - -\n");
	}
	printf("\n\n\t\t - - ENTRADA CARREGA - - \n\n");
	return 1;
}

float *aloca_matriz(int L, int C){
	float *aux = NULL;
	
	aux = (float *) calloc(L * C, sizeof(float));
	if(aux == NULL){
		printf("\n\n\t\tErro ao alocar memoria\n\n");
		exit(1);
	}else{
		return (aux);
	}
	return NULL;
}

/*
*
*VERIFICAR RETORNO
*
*/

void cal_cond_robin(){
	float keff = 0.0, numerador = 0.0, denominador = 0.0;
	float C = 1.0; // Cte adimensional que se ajusta experimentalmente C = 1.0
	
	
	//Canto superior esquerdo
	numerador = ( 2 * host_mat.perm[tam_mat_real + 1] * host_mat.perm[tam_mat_real + 2] );
	denominador = ( host_mat.perm[tam_mat_real + 1] + host_mat.perm[tam_mat_real + 2] );
	keff = numerador / denominador;
	host_b.R[tam_mat_real + 1] = C*h/keff;
	
	numerador = (2 * host_mat.perm[tam_mat_real + 1] * host_mat.perm[(2*tam_mat_real) + 1]);
	denominador = ( host_mat.perm[tam_mat_real + 1] + host_mat.perm[(2*tam_mat_real) + 1]);
	keff = numerador / denominador;
	host_b.D[tam_mat_real + 1] = C*h/keff;
	
	//Canto superior direito
	numerador = ( 2 * host_mat.perm[tam_mat_real + tam_mat_interna] * host_mat.perm[tam_mat_real + (tam_mat_interna - 1)] );
	denominador = ( host_mat.perm[tam_mat_real + tam_mat_interna] + host_mat.perm[tam_mat_real + (tam_mat_interna - 1)] );
	keff = numerador / denominador;
	host_b.L[tam_mat_real + tam_mat_interna] = C*h/keff;
	
	numerador = ( 2 * host_mat.perm[tam_mat_real + tam_mat_interna] * host_mat.perm[(2 * tam_mat_real) + tam_mat_interna] );
	denominador = ( host_mat.perm[tam_mat_real + tam_mat_interna] + host_mat.perm[(2 * tam_mat_real) + tam_mat_interna] );
	keff = numerador / denominador;
	host_b.D[tam_mat_real + tam_mat_interna] = C*h/keff;
	
	//Canto infeior esquerdo
	numerador = ( 2 * host_mat.perm[(tam_mat_real * tam_mat_interna) + 1] * host_mat.perm[(tam_mat_real * (tam_mat_interna - 1)) + 1] );
	denominador = ( host_mat.perm[(tam_mat_real * tam_mat_interna) + 1] + host_mat.perm[(tam_mat_real * (tam_mat_interna - 1)) + 1] );
	keff = numerador / denominador;
	host_b.U[(tam_mat_real * tam_mat_interna) + 1] = C*h/keff;
	
	numerador = ( 2 * host_mat.perm[(tam_mat_real * tam_mat_interna) + 1] * host_mat.perm[(tam_mat_real * tam_mat_interna) + 2] );
	denominador = ( host_mat.perm[(tam_mat_real * tam_mat_interna) + 1] + host_mat.perm[(tam_mat_real * tam_mat_interna) + 2] );
	keff = numerador / denominador;
	host_b.R[(tam_mat_real * tam_mat_interna) + 1] = C*h/keff;
	
	//Canto infeior direito
	numerador = ( 2 * host_mat.perm[(tam_mat_real * tam_mat_interna) + tam_mat_interna] * host_mat.perm[(tam_mat_real * (tam_mat_interna - 1)) + tam_mat_interna] );
	denominador = ( host_mat.perm[(tam_mat_real * tam_mat_interna) + tam_mat_interna] + host_mat.perm[(tam_mat_real * (tam_mat_interna - 1)) + tam_mat_interna] );
	keff = numerador / denominador;
	host_b.U[(tam_mat_real * tam_mat_interna) + tam_mat_interna] = C*h/keff;
	
	numerador = ( 2 * host_mat.perm[(tam_mat_real * tam_mat_interna) + tam_mat_interna] * host_mat.perm[(tam_mat_real * tam_mat_interna) + (tam_mat_interna - 1)] );
	denominador = ( host_mat.perm[(tam_mat_real * tam_mat_interna) + tam_mat_interna] + host_mat.perm[(tam_mat_real * tam_mat_interna) + (tam_mat_interna - 1)] );
	keff = numerador / denominador;
	host_b.L[(tam_mat_real * tam_mat_interna) + tam_mat_interna] = C*h/keff;

	//Calculo das fronteiras e região interna para betas
	int i = 0;
	for(i = 2; i < tam_mat_interna; i ++){
		
		//Calcula fronteira superior
		numerador = ( 2 * host_mat.perm[tam_mat_real + i] * host_mat.perm[tam_mat_real + (i-1)] );
		denominador = ( host_mat.perm[tam_mat_real + i] + host_mat.perm[tam_mat_real + (i-1)] );
		keff = numerador / denominador;
		host_b.L[tam_mat_real + i] = C*h/keff;
		
		numerador = ( 2 * host_mat.perm[tam_mat_real + i] * host_mat.perm[tam_mat_real + (i+1)] );
		denominador = ( host_mat.perm[tam_mat_real + i] + host_mat.perm[tam_mat_real + (i+1)] );
		keff = numerador / denominador;
		host_b.R[tam_mat_real + i] = C*h/keff;
		
		numerador = ( 2 * host_mat.perm[tam_mat_real + i] * host_mat.perm[(2 * tam_mat_real) + i] );
		denominador = ( host_mat.perm[tam_mat_real + i] + host_mat.perm[(2 * tam_mat_real) + i] );
		keff = numerador / denominador;
		host_b.D[tam_mat_real + i] = C*h/keff;
		
		//Calcula fronteira esquerda
		numerador = ( 2 * host_mat.perm[(i * tam_mat_real) + 1] * host_mat.perm[((i - 1) * tam_mat_real) + 1] );
		denominador = ( host_mat.perm[(i * tam_mat_real) + 1] + host_mat.perm[((i - 1) * tam_mat_real) + 1] );
		keff = numerador / denominador;
		host_b.U[(i * tam_mat_real) + 1] = C*h/keff;
		
		numerador = ( 2 * host_mat.perm[(i * tam_mat_real) + 1] * host_mat.perm[(i * tam_mat_real) + 2] );
		denominador = ( host_mat.perm[(i * tam_mat_real) + 1] + host_mat.perm[(i * tam_mat_real) + 2] );
		keff = numerador / denominador;
		host_b.R[(i * tam_mat_real) + 1] = C*h/keff;
		
		numerador = ( 2 * host_mat.perm[(i * tam_mat_real) + 1] * host_mat.perm[((i + 1) * tam_mat_real) + 1] );
		denominador = ( host_mat.perm[(i * tam_mat_real) + 1] + host_mat.perm[((i + 1) * tam_mat_real) + 1] );
		keff = numerador / denominador;
		host_b.D[(i * tam_mat_real) + 1] = C*h/keff;
		
		//Calcula fronteira inferior
		numerador = ( 2 * host_mat.perm[(tam_mat_interna * tam_mat_real) + i] * host_mat.perm[(tam_mat_interna * tam_mat_real) + (i - 1)] );
		denominador = ( host_mat.perm[(tam_mat_interna * tam_mat_real) + i] + host_mat.perm[(tam_mat_interna * tam_mat_real) + (i - 1)] );
		keff = numerador / denominador;
		host_b.L[(tam_mat_interna * tam_mat_real) + i] = C*h/keff;
		
		numerador = ( 2 * host_mat.perm[(tam_mat_interna * tam_mat_real) + i] * host_mat.perm[((tam_mat_interna - 1) * tam_mat_real) + i] );
		denominador = ( host_mat.perm[(tam_mat_interna * tam_mat_real) + i] + host_mat.perm[((tam_mat_interna - 1) * tam_mat_real) + i] );
		keff = numerador / denominador;
		host_b.U[(tam_mat_interna * tam_mat_real) + i] = C*h/keff;
		
		numerador = ( 2 * host_mat.perm[(tam_mat_interna * tam_mat_real) + i] * host_mat.perm[(tam_mat_interna * tam_mat_real) + (i + 1)] );
		denominador = ( host_mat.perm[(tam_mat_interna * tam_mat_real) + i] + host_mat.perm[(tam_mat_interna * tam_mat_real) + (i + 1)] );
		keff = numerador / denominador;
		host_b.R[(tam_mat_interna * tam_mat_real) + i] = C*h/keff;
		
		//Calcula fronteira direita
		numerador = ( 2 * host_mat.perm[(i * tam_mat_real) + tam_mat_interna] * host_mat.perm[((i-1) * tam_mat_real) + tam_mat_interna] );
		denominador = ( host_mat.perm[(i * tam_mat_real) + tam_mat_interna] + host_mat.perm[((i-1) * tam_mat_real) + tam_mat_interna] );
		keff = numerador / denominador;
		host_b.U[(i * tam_mat_real) + tam_mat_interna] = C*h/keff;
		
		numerador = ( 2 * host_mat.perm[(i * tam_mat_real) + tam_mat_interna] * host_mat.perm[(i * tam_mat_real) + (tam_mat_interna - 1)] );
		denominador = ( host_mat.perm[(i * tam_mat_real) + tam_mat_interna] + host_mat.perm[(i * tam_mat_real) + (tam_mat_interna - 1)] );
		keff = numerador / denominador;
		host_b.L[(i * tam_mat_real) + tam_mat_interna] = C*h/keff;
		
		numerador = ( 2 * host_mat.perm[(i * tam_mat_real) + tam_mat_interna] * host_mat.perm[((i+1) * tam_mat_real) + tam_mat_interna] );
		denominador = ( host_mat.perm[(i * tam_mat_real) + tam_mat_interna] + host_mat.perm[((i+1) * tam_mat_real) + tam_mat_interna] );
		keff = numerador / denominador;
		host_b.D[(i * tam_mat_real) + tam_mat_interna] = C*h/keff;
		
		//Calcula dados internos
		int j = 0;
		for(j = 2; j < tam_mat_interna; j ++){
			numerador = ( 2 * host_mat.perm[(i * tam_mat_real) + j] * host_mat.perm[(i * tam_mat_real) + (j - 1)] );
			denominador = ( host_mat.perm[(i * tam_mat_real) + j] + host_mat.perm[(i * tam_mat_real) + (j - 1)] );
			keff = numerador / denominador;
			host_b.L[(i * tam_mat_real) + j] = C*h/keff;
		
			numerador = ( 2 * host_mat.perm[(i * tam_mat_real) + j] * host_mat.perm[(i * tam_mat_real) + (j + 1)] );
			denominador = ( host_mat.perm[(i * tam_mat_real) + j] + host_mat.perm[(i * tam_mat_real) + (j + 1)] );
			keff = numerador / denominador;
			host_b.R[(i * tam_mat_real) + j] = C*h/keff;
		
			numerador = ( 2 * host_mat.perm[(i * tam_mat_real) + j] * host_mat.perm[((i - 1) * tam_mat_real) + j] );
			denominador = ( host_mat.perm[(i * tam_mat_real) + j] + host_mat.perm[((i - 1) * tam_mat_real) + j] );
			keff = numerador / denominador;
			host_b.U[(i * tam_mat_real) + j] = C*h/keff;
		
			numerador = ( 2 * host_mat.perm[(i * tam_mat_real) + j] * host_mat.perm[((i + 1) * tam_mat_real) + j] );
			denominador = ( host_mat.perm[(i * tam_mat_real) + j] + host_mat.perm[((i + 1) * tam_mat_real) + j] );
			keff = numerador / denominador;
			host_b.D[(i * tam_mat_real) + j] = C*h/keff;
		}
	}
}

/*
*
*VERIFICAR RETORNO
*
*/

char parametro_independentes(){
	int i = 0, j = 0;
	float constante = 2/h;
	
	for(i = 0; i < tam_mat_real; i ++)
		for(j = 0; j < tam_mat_real; j++){
			host_mat.epsilon[i*tam_mat_real + j] = constante * host_mat.perm[i*tam_mat_real + j];
			host_mat.font[i*tam_mat_real + j] *= h;
		}
	
	return 0;
}


char copia_dados_para_gpu(){
	
	HANDLE_ERROR( cudaMemcpy( dev_q.R, host_q.R, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_q.L, host_q.L, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_q.U, host_q.U, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_q.D, host_q.D, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	
	HANDLE_ERROR( cudaMemcpy( dev_q.R_old, host_q.R_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_q.L_old, host_q.L_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_q.U_old, host_q.U_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_q.D_old, host_q.D_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	
	
	HANDLE_ERROR( cudaMemcpy( dev_l.R, host_l.R, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_l.L, host_l.L, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_l.U, host_l.U, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_l.D, host_l.D, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	
	HANDLE_ERROR( cudaMemcpy( dev_l.R_old, host_l.R_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_l.L_old, host_l.L_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_l.U_old, host_l.U_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_l.D_old, host_l.D_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	
	
	HANDLE_ERROR( cudaMemcpy( dev_b.R, host_b.R, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b.L, host_b.L, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b.U, host_b.U, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b.D, host_b.D, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	
	HANDLE_ERROR( cudaMemcpy( dev_b.R_old, host_b.R_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b.L_old, host_b.L_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b.U_old, host_b.U_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b.D_old, host_b.D_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	
	HANDLE_ERROR( cudaMemcpy( dev_pressao.p, host_pressao.p, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_pressao.p_old, host_pressao.p_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	
	HANDLE_ERROR( cudaMemcpy( dev_mat.perm, host_mat.perm, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_mat.epsilon, host_mat.epsilon, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_mat.font, host_mat.font, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyHostToDevice ) );
	
	return 0;
}

void copia_dados_para_cpu(){
	
	HANDLE_ERROR( cudaMemcpy( host_q.R, dev_q.R, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_q.L, dev_q.L, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_q.U, dev_q.U, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_q.D, dev_q.D, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	
	HANDLE_ERROR( cudaMemcpy( host_q.R_old, dev_q.R_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_q.L_old, dev_q.L_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_q.U_old, dev_q.U_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_q.D_old, dev_q.D_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	
	
	HANDLE_ERROR( cudaMemcpy( host_l.R, dev_l.R, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_l.L, dev_l.L, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_l.U, dev_l.U, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_l.D, dev_l.D, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	
	HANDLE_ERROR( cudaMemcpy( host_l.R_old, dev_l.R_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_l.L_old, dev_l.L_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_l.U_old, dev_l.U_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_l.D_old, dev_l.D_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	
	
	HANDLE_ERROR( cudaMemcpy( host_b.R, dev_b.R, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_b.L, dev_b.L, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_b.U, dev_b.U, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_b.D, dev_b.D, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	
	HANDLE_ERROR( cudaMemcpy( host_b.R_old, dev_b.R_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_b.L_old, dev_b.L_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_b.U_old, dev_b.U_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_b.D_old, dev_b.D_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	
	HANDLE_ERROR( cudaMemcpy( host_pressao.p, dev_pressao.p, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_pressao.p_old, dev_pressao.p_old, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	
	HANDLE_ERROR( cudaMemcpy( host_mat.font, dev_mat.font, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_mat.perm, dev_mat.perm, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( host_mat.epsilon, dev_mat.epsilon, tam_mat_real * tam_mat_real * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
}

char inicializa_parametros(){
	printf("\n\n\t\t- - INICIALIZANDO PARAMETROS - - \n\n\n");
	
					/*
					 * 
					 * 
					 * CONTRUIR FUNCAO PARA VERIFICAR ERRO DE ALOCAÇÃO
					 * VERIFICAR RETORNO
					 */
	
	tam_mat_real = tam_mat_interna + 2;
	h = tam_regiao / tam_mat_interna;
	
	HANDLE_ERROR( cudaMalloc( (void**)&dev_q, sizeof(ESTRUTURA_Q) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_l, sizeof(ESTRUTURA_L) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_b, sizeof(ESTRUTURA_B) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_pressao, sizeof(ESTRUTURA_PRESSAO) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_mat, sizeof(ESTRUTURA_MAT) ) );
	
	host_q.R = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_q.R != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q.R, tam_mat_real * tam_mat_real * sizeof(float) ) );
		
	host_q.L = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_q.L != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q.L, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	host_q.U = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_q.U != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q.U, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	host_q.D = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_q.D != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q.D, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	host_q.R_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_q.R_old != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q.R_old, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	host_q.L_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_q.L_old != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q.L_old, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	host_q.U_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_q.U_old != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q.U_old, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	host_q.D_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_q.D_old != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q.D_old, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	host_l.R = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_l.R != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l.R, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	host_l.L = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_l.L != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l.L, tam_mat_real * tam_mat_real * sizeof(float) ) );
		
	host_l.U = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_l.U != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l.U, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	host_l.D = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_l.D != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l.D, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	host_l.R_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_l.R_old != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l.R_old, tam_mat_real * tam_mat_real * sizeof(float) ) );
		
	host_l.L_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_l.L_old != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l.L_old, tam_mat_real * tam_mat_real * sizeof(float) ) );
		
	host_l.U_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_l.U_old != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l.U_old, tam_mat_real * tam_mat_real * sizeof(float) ) );
		
	host_l.D_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_l.D_old != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l.D_old, tam_mat_real * tam_mat_real * sizeof(float) ) );
		
	host_b.R = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_b.R != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b.R, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	host_b.L = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_b.L != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b.L, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	host_b.U = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_b.U != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b.U, tam_mat_real * tam_mat_real * sizeof(float) ) );
		
	host_b.D = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_b.D != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b.D, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	host_b.R_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_b.R_old != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b.R_old, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	host_b.L_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_b.L_old != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b.L_old, tam_mat_real * tam_mat_real * sizeof(float) ) );
		
	host_b.U_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_b.U_old != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b.U_old, tam_mat_real * tam_mat_real * sizeof(float) ) );
		
	host_b.D_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_b.D_old != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b.D_old, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	host_pressao.p = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_pressao.p != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_pressao.p, tam_mat_real * tam_mat_real * sizeof(float) ) );
		
	host_pressao.p_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(host_pressao.p_old != NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_pressao.p_old, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	
	HANDLE_ERROR( cudaMalloc( (void**)&dev_mat.perm, tam_mat_real * tam_mat_real * sizeof(float) ) );


	HANDLE_ERROR( cudaMalloc( (void**)&dev_mat.font, tam_mat_real * tam_mat_real * sizeof(float) ) );


	HANDLE_ERROR( cudaMalloc( (void**)&dev_mat.epsilon, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	HANDLE_ERROR( cudaMalloc( (void**)&dev_aux, tam_mat_real * tam_mat_real * sizeof(float) ) );
	HANDLE_ERROR( cudaMemset( dev_aux, 0, tam_mat_real * tam_mat_real * sizeof(float) ) );
	
	HANDLE_ERROR( cudaMalloc( (void**)&erro_max, sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_erro, sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_media, tam_mat_real * tam_mat_real * sizeof(float) ) );
	int i = 0;
	switch(op_contorno){
		case 1: //Inicializa contorno superior
			for(i = 0; i < tam_mat_real; i++){
				host_q.D[i] = valor_contor;
				host_q.D_old[i] = valor_contor;
			}
			break;
			
		case 2://Inicializa contorno esquerdo
			for(i = 0; i < tam_mat_real; i++){
				host_q.R[i*tam_mat_real] = valor_contor;
				host_q.R_old[i*tam_mat_real] = valor_contor;
			}
			break;
			
		case 3://Inicializa contorno direito
			for(i = 0; i < tam_mat_real; i++){
				host_q.L[i*tam_mat_real + (tam_mat_real - 1)] = valor_contor;
				host_q.L_old[i*tam_mat_real + (tam_mat_real - 1)] = valor_contor;
			}
			break;
			
		case 4://Inicializa contorno inferior
			for(i = 0; i < tam_mat_real; i++){
				host_q.L[(tam_mat_real-1)*tam_mat_real + i] = valor_contor;
				host_q.L_old[(tam_mat_real-1)*tam_mat_real + i] = valor_contor;
			}
			break;
			
		default:
			printf("\n\n\t\t - - OCORREU ALGUM ERRO NA OPCAO DE CONTORNO - - \n\n");
			break;
	}
	printf("\n\n\t\t- - FIM DA INICIALIZACAO PARAMETROS - - \n\n\n");
	return 1;
}

void clear_mem(){
	HANDLE_ERROR( cudaFree (dev_q.U));
	HANDLE_ERROR( cudaFree (dev_q.R));
	HANDLE_ERROR( cudaFree (dev_q.D));
	HANDLE_ERROR( cudaFree (dev_q.L));
	
	free(host_q.U);
	free(host_q.R);
	free(host_q.D);
	free(host_q.L);
	
	HANDLE_ERROR( cudaFree (dev_l.U));
	HANDLE_ERROR( cudaFree (dev_l.R));
	HANDLE_ERROR( cudaFree (dev_l.D));
	HANDLE_ERROR( cudaFree (dev_l.L));
	
	free(host_l.U);
	free(host_l.R);
	free(host_l.D);
	free(host_l.L);
	
	HANDLE_ERROR( cudaFree (dev_b.U));
	HANDLE_ERROR( cudaFree (dev_b.R));
	HANDLE_ERROR( cudaFree (dev_b.D));
	HANDLE_ERROR( cudaFree (dev_b.L));
	
	free(host_b.U);
	free(host_b.R);
	free(host_b.D);
	free(host_b.L);
	
	HANDLE_ERROR( cudaFree (dev_pressao.p));
	HANDLE_ERROR( cudaFree (dev_pressao.p_old));
	
	free(host_pressao.p);
	free(host_pressao.p_old);
	
	HANDLE_ERROR( cudaFree (dev_mat.perm));
	HANDLE_ERROR( cudaFree (dev_mat.font));
	HANDLE_ERROR( cudaFree (dev_mat.epsilon));
	
	free(host_mat.perm);
	free(host_mat.font);
	free(host_mat.epsilon);
}