#include <stdio.h>
#include <stdlib.h>
#include  <string.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define N       100
#define DIM      2

char le_entrada();
char inicializa_parametros();
double * aloca_matriz(int, int);
void cal_cond_robin();
char parametro_independentes();
char copia_dados_para_gpu();
void copia_dados_para_cpu();
char calcula_pressao_velocidade(int, int, int, int, int);
char atualiza_mult_lagrange(int tid);

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

/* - - - - - - - Entradas Externas  - - - - - - - */
int tam_mat_interna = 3, tam_mat_real = 3 + 2, max_interacoes = 1000, op_contorno = 1;
double tam_regiao = 20000.00, erro_max = 1e-5, valor_contor = 2.00;
double h = 20000.00 / 3; // ALTURA H = TAM_REGIAO / TAM_MAT_INTERNA
double *mat_perm = NULL, *mat_font = NULL, *mat_epsilon = NULL;
/* - - - - - - - Fim das Entradas Externas  - - - - - - - */




/* - - - - - - - Ponteiros para CPU  - - - - - - - */
double *q_R = NULL, *q_L = NULL, *q_U = NULL, *q_D = NULL;
double *q_R_old = NULL, *q_L_old = NULL, *q_U_old = NULL, *q_D_old = NULL;

double *l_R = NULL, *l_L = NULL, *l_U = NULL, *l_D = NULL;
double *l_R_old = NULL, *l_L_old = NULL, *l_U_old = NULL, *l_D_old = NULL;

double *b_R = NULL, *b_L = NULL, *b_U = NULL, *b_D = NULL;
double *b_R_old = NULL, *b_L_old = NULL, *b_U_old = NULL, *b_D_old = NULL;

double *pressao = NULL, *pressao_old = NULL;


/* - - - - - - - Ponteiros para GPU  - - - - - - - */
double *dev_mat_perm = NULL, *dev_mat_font = NULL, *dev_mat_epsilon = NULL;


double *dev_q_R = NULL, *dev_q_L = NULL, *dev_q_U = NULL, *dev_q_D = NULL;
double *dev_q_R_old = NULL, *dev_q_L_old = NULL, *dev_q_U_old = NULL, *dev_q_D_old = NULL;

double *dev_l_R = NULL, *dev_l_L = NULL, *dev_l_U = NULL, *dev_l_D = NULL;
double *dev_l_R_old = NULL, *dev_l_L_old = NULL, *dev_l_U_old = NULL, *dev_l_D_old = NULL;

double *dev_b_R = NULL, *dev_b_L = NULL, *dev_b_U = NULL, *dev_b_D = NULL;
double *dev_b_R_old = NULL, *dev_b_L_old = NULL, *dev_b_U_old = NULL, *dev_b_D_old = NULL;

double *dev_pressao = NULL, *dev_pressao_old = NULL;



//- - - - - - - - - - - - - - FIM - GLOBAIS - - - - - - - - - - - - - - //



__global__ void escoamento_monofasico(int *a){
    /*int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

        a[offset] = offset;*/
	/*vificar as condições de contorno*/	
		
		
		
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
	int eq_tid_cant_sup_dir = blockDim.x * gridDim.x  - 1; // posição extremo sup direito
	int eq_tid_cant_inf_dir = ((gridDim.x * blockDim.x) * (gridDim.y * blockDim.y)) - 1; // posição extremo inf direito
	int eq_tid_cant_inf_esq = (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y - 1); // posição extremo inf esquerdo
	double media = 0.0, sum1 = 0.0, sum2 = 0.0;


	if(tid == 0){//canto superior esquerdo
		a[tid] = 1;
		/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
		calcula_pressao_velocidade(tid, 0, 1, 1, 0);
		atualiza_mult_lagrange(tid);
		
		flag_thread_centrais = 0;
	}
	
	if(tid == eq_tid_cant_sup_dir){//canto superior direito
		a[tid] = 3;
		/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
		calcula_pressao_velocidade(tid, 0, 0, 1, 1);
		atualiza_mult_lagrange(tid);
		
		flag_thread_centrais = 0;
	}
	
	if(tid == eq_tid_cant_inf_esq){//canto inferior esquerdo
		a[tid] = 7;  
		/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
		calcula_pressao_velocidade(tid, 0, 1, 1, 0);
		atualiza_mult_lagrange(tid);
		
		flag_thread_centrais = 0;
	}
	
	if(tid == eq_tid_cant_inf_dir){//canto inferior direito
		a[tid] = 9;
		/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
		calcula_pressao_velocidade(tid, 1, 0, 0, 1);
		atualiza_mult_lagrange(tid);
		
		flag_thread_centrais = 0;
	}
	
	if((tid > 0) && (tid < eq_tid_cant_sup_dir)){//fronteira superior
		a[tid] = 2;
		/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
		calcula_pressao_velocidade(tid, 0, 1, 1, 1);
		atualiza_mult_lagrange(tid);
		
		flag_thread_centrais = 0;
	}
	
	if((tid > eq_tid_cant_sup_dir) && (tid < eq_tid_cant_inf_dir) && (tid % dimensao_x == eq_tid_cant_sup_dir)){ //fronteira direita
		a[tid] = 6;
		/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
		calcula_pressao_velocidade(tid, 1, 0, 1, 1);
		atualiza_mult_lagrange(tid);
		
		flag_thread_centrais = 0;
	}
	
	if((tid > eq_tid_cant_inf_esq) && (tid < eq_tid_cant_inf_dir)){ //fronteira inferior
		a[tid] = 8;
		/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
		calcula_pressao_velocidade(tid, 1, 1, 0, 1);
		atualiza_mult_lagrange(tid);
		
		flag_thread_centrais = 0;
	}
	
	if((tid > 0) && (tid < eq_tid_cant_inf_dir) && (tid < eq_tid_cant_inf_esq) && (tid % dimensao_y == 0)){//fronteira esquerda
		a[tid] = 4;
		/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
		calcula_pressao_velocidade(tid, 1, 1, 1, 0);
		atualiza_mult_lagrange(tid);
		
		flag_thread_centrais = 0;
	}
	
	if(flag_thread_centrais){
		a[tid] = 5;
		/*VERIFICAR AS CONDIÇÕES DE CONTORNO*/
		calcula_pressao_velocidade(tid, 1, 1, 1, 1);
		atualiza_mult_lagrange(tid);
	}
	
	/*
	*
	*SINCRONIZA
	*COMENTARIOS 
		*ALOCAR VARIÁVEL aux com o tamanho de "tids"
		*VERIFICAR ATOMICIDADE PRA VALORES FLOAT
		*VERIFICAR ALOCAÇÃO DAS MEMÓRIAS GLOBAIS
		*alocar memória erro
	*/
	/* 
	 * Imponiendo a media cero na distribuicao de presiones
	 * Calculo de la media
	 */
	
	atomicAdd( &media, dev_pressao[tid] );
	//atomicSub( &aux[tid], dev_pressao[tid] - dev_pressao_old[tid] );
	__syncthreads();
	
	dev_pressao[tid] -= M;
	dev_l_D[tid] -= M;
	dev_l_U[tid] -= M;
	dev_l_L[tid] -= M;
	dev_l_R[tid] -= M;
	
	/*avaliando criterio de convergencia*/
	aux[tid] = dev_pressao[tid] - dev_b_D_old[tid];
	__syncthreads();
	
	atomicAdd( &sum1, aux[tid] * aux[tid] );
	atomicAdd( &sum2, dev_pressao[tid] * dev_pressao[tid] );
	__syncthreads();
	if(tid == 0)
		erro = sqrt(sum1/sum2);
		
	if (erro < 1e-5) return 0;
	
	p_old[j][k] = p[j][k];
	dev_pressao_old[tid] = dev_pressao_old[tid];
	dev_q_U_old[tid] = dev_q_U[tid];
	dev_q_R_old[tid] = dev_q_R[tid];
	dev_q_L_old[tid] = dev_q_L[tid];
	dev_q_D_old[tid] = dev_q_D[tid];
	
	dev_l_D_old[tid] = dev_l_D[tid];
	dev_l_U_old[tid] = dev_l_U[tid];
	dev_l_L_old[tid] = dev_l_L[tid];
	dev_l_R_old[tid] = dev_l_R[tid];
}
__device__ char atualiza_mult_lagrange(int tid){
	int index_mem_central = 0, index_mem_down = 0, index_mem_uper = 0;
	int index_mem_left = 0, index_mem_right = 0;
	int comprimento_kernel = blockDim.x * gridDim.x;
	int offset = (blockDim.x * gridDim.x) + 1;
	
	index_mem_central = tid + ((tid/comprimento_kernel)*2) + offset;
	index_mem_uper = index_mem_central - (offset -1); // (offset -1) = comprimento do kernel
	index_mem_down = index_mem_central + (offset -1);
	index_mem_left = index_mem_central - 1;
	index_mem_right = index_mem_central + 1;
	
	dev_l_U[index_mem_central] = dev_b_U[index_mem_central] * (dev_q_U[index_mem_central] + dev_q_D_old[index_mem_uper]) + dev_l_D_old[index_mem_uper];
	dev_l_D[index_mem_central] = dev_b_D[index_mem_central] * (dev_q_D[index_mem_central] + dev_q_U_old[index_mem_down]) + dev_l_U_old[index_mem_down];
	dev_l_R[index_mem_central] = dev_b_R[index_mem_central] * (dev_q_R[index_mem_central] + dev_q_L_old[index_mem_right]) + dev_l_L_old[index_mem_right];
	dev_l_L[index_mem_central] = dev_b_L[index_mem_central] * (dev_q_L[index_mem_central] + dev_q_R_old[index_mem_left]) + dev_l_R_old[index_mem_left];
	
}

__device__ char calcula_pressao_velocidade(int tid, int uper, int right, int down, int left){
	double auxU = 0.0, auxD = 0.0, auxR = 0.0, auxL = 0.0, DU = 0.0, DD = 0.0, DR = 0.0, DL = 0.0;
	int index_mem_central = 0, index_mem_down = 0, index_mem_uper = 0;
	int index_mem_left = 0, index_mem_right = 0;
	int comprimento_kernel = blockDim.x * gridDim.x;
	int offset = (blockDim.x * gridDim.x) + 1;
	
	index_mem_central = tid + ((tid/comprimento_kernel)*2) + offset;
	index_mem_uper = index_mem_central - (offset -1); // (offset -1) = comprimento do kernel
	index_mem_down = index_mem_central + (offset -1);
	index_mem_left = index_mem_central - 1;
	index_mem_right = index_mem_central + 1;
	
	if(uper == 1){
		auxU = dev_mat_epsilon[index_mem_central] / (1 + dev_b_U[index_mem_central] * dev_mat_epsilon[index_mem_central]);
		DU = auxU * (dev_b_U[index_mem_central] * dev_q_D_old[index_mem_uper] + dev_l_D_old[index_mem_uper]);
	}
	
	if(right == 1){
		auxR = dev_mat_epsilon[index_mem_central] / (1 + dev_b_R[index_mem_central] * dev_mat_epsilon[index_mem_central]);
		DR = auxR * (dev_b_R[index_mem_central] * dev_q_L_old[index_mem_right] + dev_l_L_old[index_mem_right]);
	}
	
	if(down == 1){
		auxD = dev_mat_epsilon[index_mem_central] / (1 + dev_b_D[index_mem_central] * dev_mat_epsilon[index_mem_central]);
		DD = auxD * (dev_b_D[index_mem_central] * dev_q_U_old[index_mem_down] + dev_l_U_old[index_mem_down]);
	}
	
	if(left == 1){
		auxL = dev_mat_epsilon[index_mem_central] / (1 + dev_b_L[index_mem_central] * dev_mat_epsilon[index_mem_central]);
		DL = auxL * (dev_b_L[index_mem_central] * dev_q_R_old[index_mem_left] + dev_l_R_old[index_mem_left]);
	}
	
	dev_pressao[index_mem_central] = (dev_mat_font[index_mem_central] + DU + DR + DD + DL) / (auxU + auxR + auxD + auxL);
	
	dev_q_L[index_mem_central] = auxL * dev_pressao[index_mem_central] - DL;
	dev_q_R[index_mem_central] = auxR * dev_pressao[index_mem_central] - DR;
	dev_q_U[index_mem_central] = auxU * dev_pressao[index_mem_central] - DU;
	dev_q_D[index_mem_central] = auxD * dev_pressao[index_mem_central] - DD;
	
	return 0;
}


int main(void){
	le_entrada();
	inicializa_parametros();
	cal_cond_robin();
	parametro_independentes();
	int i = 0, j = 0;
	/*
	printf("\ntam_mat_interna = %d\n", tam_mat_interna);
	printf("tam_mat_real = %d\n", tam_mat_real);
	printf("max_interacoes = %d\n", max_interacoes);
	printf("op_contorno = %d\n", op_contorno);
	printf("tam_regiao = %lf\n", tam_regiao);
	printf("erro_max = %lf\n", erro_max);
	printf("valor_contor = %lf\n", valor_contor);
	
	printf("\n\nmat_font:\n");
	
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%1.e ", mat_font[i*tam_mat_real + j]);
		printf("\n");
	}
	
	
	printf("\n\nmat_perm:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%1.e ", mat_perm[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\nmat_epsilon:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%1.e ", mat_epsilon[i*tam_mat_real + j]);
		printf("\n");
	}
	*/
	printf("\n\nb_U:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%1.e ", b_U[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\nb_R:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%1.e ", b_R[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\nb_D:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%1.e ", b_D[i*tam_mat_real + j]);
		printf("\n");
	}
	
	printf("\n\nb_L:\n");
	for(i = 0; i < tam_mat_real; i ++){
		for(j = 0; j < tam_mat_real; j++)
			printf("%1.e ", b_L[i*tam_mat_real + j]);
		printf("\n");
	}
	
	system("pause");
	return 0;
}
 
 char le_entrada(){
	printf("\n\n\t\t - - CARREGANDO ENTRADA - - \n\n");
	FILE *arq = NULL;
	
	arq = fopen("../dir_entrada/parametro_entrada.txt", "r");
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
				
					fscanf(arq, "%lf", &erro_max);
					break;
					
				case 10: //tam_regiao
				
					fscanf(arq, "%lf", &tam_regiao);
					break;
					
				case 11: //opcao_contorno
				
					fscanf(arq, "%d", &op_contorno);
					break;
					
				case 12: //valor_contor
				
					fscanf(arq, "%lf", &valor_contor);
					break;
					
				case 14: //max_interacoes
				
					fscanf(arq, "%d", &max_interacoes);
					break;
					
				case 15: //tam_mat_interna
				
					fscanf(arq, "%d", &tam_mat_interna);
					break;
					
				case 16: //matriz_de_fontes
					//uso (tam_mat_interna + 2) - pois ainda não inicializei 'tam_mat_real'
					mat_font = aloca_matriz(tam_mat_interna + 2, tam_mat_interna + 2);
					for(i = 1; i < (tam_mat_interna + 2) - 1; i ++)
						for(j = 1; j < (tam_mat_interna + 2) - 1 ; j++)
							fscanf(arq, "%lf", &mat_font[i*(tam_mat_interna+2) + j]);
							
					break;
					
				case 18: //matriz_permeabilidade
				
					mat_perm = aloca_matriz(tam_mat_interna + 2, tam_mat_interna + 2);
					mat_epsilon = aloca_matriz(tam_mat_interna + 2, tam_mat_interna + 2);
					for(i = 1; i < (tam_mat_interna + 2) - 1; i ++)
						for(j = 1; j < (tam_mat_interna + 2) - 1 ; j++)
							fscanf(arq, "%lf", &mat_perm[i*(tam_mat_interna+2) + j]);
							
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

char inicializa_parametros(){
	printf("\n\n\t\t- - INICIALIZANDO PARAMETROS - - \n\n\n");
	
	tam_mat_real = tam_mat_interna + 2;
	h = tam_regiao / tam_mat_interna;
	
	q_R = aloca_matriz(tam_mat_real, tam_mat_real);
	if(q_R == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q_R, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	q_L = aloca_matriz(tam_mat_real, tam_mat_real);
	if(q_L == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q_L, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	q_U = aloca_matriz(tam_mat_real, tam_mat_real);
	if(q_U == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q_U, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	q_D = aloca_matriz(tam_mat_real, tam_mat_real);
	if(q_D == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q_D, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	q_R_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(q_R_old == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q_R_old, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	q_L_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(q_L_old == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q_L_old, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	q_U_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(q_U_old == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q_U_old, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	q_D_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(q_D_old == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_q_D_old, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	l_R = aloca_matriz(tam_mat_real, tam_mat_real);
	if(l_R == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l_R, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	l_L = aloca_matriz(tam_mat_real, tam_mat_real);
	if(l_L == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l_L, tam_mat_real * tam_mat_real * sizeof(double) ) );
		
	l_U = aloca_matriz(tam_mat_real, tam_mat_real);
	if(l_U == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l_U, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	l_D = aloca_matriz(tam_mat_real, tam_mat_real);
	if(l_D == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l_D, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	l_R_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(l_R_old == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l_R_old, tam_mat_real * tam_mat_real * sizeof(double) ) );
		
	l_L_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(l_L_old == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l_L_old, tam_mat_real * tam_mat_real * sizeof(double) ) );
		
	l_U_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(l_U_old == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l_U_old, tam_mat_real * tam_mat_real * sizeof(double) ) );
		
	l_D_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(l_D_old == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_l_D_old, tam_mat_real * tam_mat_real * sizeof(double) ) );
		
	b_R = aloca_matriz(tam_mat_real, tam_mat_real);
	if(b_R == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b_R, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	b_L = aloca_matriz(tam_mat_real, tam_mat_real);
	if(b_L == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b_L, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	b_U = aloca_matriz(tam_mat_real, tam_mat_real);
	if(b_U == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b_U, tam_mat_real * tam_mat_real * sizeof(double) ) );
		
	b_D = aloca_matriz(tam_mat_real, tam_mat_real);
	if(b_D == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b_D, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	b_R_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(b_R_old == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b_R_old, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	b_L_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(b_L_old == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b_L_old, tam_mat_real * tam_mat_real * sizeof(double) ) );
		
	b_U_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(b_U_old == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b_U_old, tam_mat_real * tam_mat_real * sizeof(double) ) );
		
	b_D_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(b_D_old == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_b_D_old, tam_mat_real * tam_mat_real * sizeof(double) ) );
	
	pressao = aloca_matriz(tam_mat_real, tam_mat_real);
	if(pressao == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_pressao, tam_mat_real * tam_mat_real * sizeof(double) ) );
		
	pressao_old = aloca_matriz(tam_mat_real, tam_mat_real);
	if(pressao_old == NULL)
		HANDLE_ERROR( cudaMalloc( (void**)&dev_pressao_old, tam_mat_real * tam_mat_real * sizeof(double) ) );
		
	int i = 0;
	switch(op_contorno){
		case 1: //Inicializa contorno superior
			for(i = 0; i < tam_mat_real -1; i++){
				q_D[i] = valor_contor;
				q_D_old[i] = valor_contor;
			}
			break;
			
		case 2://Inicializa contorno esquerdo
			for(i = 0; i < tam_mat_real; i++){
				q_R[i*tam_mat_real] = valor_contor;
				q_R_old[i*tam_mat_real] = valor_contor;
			}
			break;
			
		case 3://Inicializa contorno direito
			for(i = 0; i < tam_mat_real; i++){
				q_L[i*tam_mat_real + (tam_mat_real - 1)] = valor_contor;
				q_L_old[i*tam_mat_real + (tam_mat_real - 1)] = valor_contor;
			}
			break;
			
		case 4://Inicializa contorno inferior
			for(i = 0; i < tam_mat_real; i++){
				q_L[(tam_mat_real-1)*tam_mat_real + i] = valor_contor;
				q_L_old[(tam_mat_real-1)*tam_mat_real + i] = valor_contor;
			}
			break;
			
		default:
			printf("\n\n\t\t - - OCORREU ALGUM ERRO NA OPCAO DE CONTORNO - - \n\n");
			break;
	}
	printf("\n\n\t\t- - FIM DA INICIALIZACAO PARAMETROS - - \n\n\n");
	return 1;
}

double * aloca_matriz(int L, int C){
	double *aux = NULL;
	
	aux = (double *) calloc(L * C, sizeof(double));
	if(aux == NULL){
		printf("\n\n\t\tErro ao alocar memoria\n\n");
		exit(1);
	}else{
		return aux;
	}
	return NULL;
}

/*
*
*VERIFICAR RETORNO
*
*/

void cal_cond_robin(){
	double keff = 0.0, numerador = 0.0, denominador = 0.0;
	double C = 1.0; // Cte adimensional que se ajusta experimentalmente C = 1.0
	
	
	//Canto superior esquerdo
	numerador = ( 2 * mat_perm[tam_mat_real + 1] * mat_perm[tam_mat_real + 2] );
	denominador = ( mat_perm[tam_mat_real + 1] + mat_perm[tam_mat_real + 2] );
	keff = numerador / denominador;
	b_R[tam_mat_real + 1] = C*h/keff;
	
	numerador = (2 * mat_perm[tam_mat_real + 1] * mat_perm[(2*tam_mat_real) + 1]);
	denominador = ( mat_perm[tam_mat_real + 1] + mat_perm[(2*tam_mat_real) + 1]);
	keff = numerador / denominador;
	b_D[tam_mat_real + 1] = C*h/keff;
	
	//Canto superior direito
	numerador = ( 2 * mat_perm[tam_mat_real + tam_mat_interna] * mat_perm[tam_mat_real + (tam_mat_interna - 1)] );
	denominador = ( mat_perm[tam_mat_real + tam_mat_interna] + mat_perm[tam_mat_real + (tam_mat_interna - 1)] );
	keff = numerador / denominador;
	b_L[tam_mat_real + tam_mat_interna] = C*h/keff;
	
	numerador = ( 2 * mat_perm[tam_mat_real + tam_mat_interna] * mat_perm[(2 * tam_mat_real) + tam_mat_interna] );
	denominador = ( mat_perm[tam_mat_real + tam_mat_interna] + mat_perm[(2 * tam_mat_real) + tam_mat_interna] );
	keff = numerador / denominador;
	b_D[tam_mat_real + tam_mat_interna] = C*h/keff;
	
	//Canto infeior esquerdo
	numerador = ( 2 * mat_perm[(tam_mat_real * tam_mat_interna) + 1] * mat_perm[(tam_mat_real * (tam_mat_interna - 1)) + 1] );
	denominador = ( mat_perm[(tam_mat_real * tam_mat_interna) + 1] + mat_perm[(tam_mat_real * (tam_mat_interna - 1)) + 1] );
	keff = numerador / denominador;
	b_U[(tam_mat_real * tam_mat_interna) + 1] = C*h/keff;
	
	numerador = ( 2 * mat_perm[(tam_mat_real * tam_mat_interna) + 1] * mat_perm[(tam_mat_real * tam_mat_interna) + 2] );
	denominador = ( mat_perm[(tam_mat_real * tam_mat_interna) + 1] + mat_perm[(tam_mat_real * tam_mat_interna) + 2] );
	keff = numerador / denominador;
	b_R[(tam_mat_real * tam_mat_interna) + 1] = C*h/keff;
	
	//Canto infeior direito
	numerador = ( 2 * mat_perm[(tam_mat_real * tam_mat_interna) + tam_mat_interna] * mat_perm[(tam_mat_real * (tam_mat_interna - 1)) + tam_mat_interna] );
	denominador = ( mat_perm[(tam_mat_real * tam_mat_interna) + tam_mat_interna] + mat_perm[(tam_mat_real * (tam_mat_interna - 1)) + tam_mat_interna] );
	keff = numerador / denominador;
	b_U[(tam_mat_real * tam_mat_interna) + tam_mat_interna] = C*h/keff;
	
	numerador = ( 2 * mat_perm[(tam_mat_real * tam_mat_interna) + tam_mat_interna] * mat_perm[(tam_mat_real * tam_mat_interna) + (tam_mat_interna - 1)] );
	denominador = ( mat_perm[(tam_mat_real * tam_mat_interna) + tam_mat_interna] + mat_perm[(tam_mat_real * tam_mat_interna) + (tam_mat_interna - 1)] );
	keff = numerador / denominador;
	b_L[(tam_mat_real * tam_mat_interna) + tam_mat_interna] = C*h/keff;

	//Calculo das fronteiras e região interna para betas
	int i = 0;
	for(i = 2; i < tam_mat_interna; i ++){
		
		//Calcula fronteira superior
		numerador = ( 2 * mat_perm[tam_mat_real + i] * mat_perm[tam_mat_real + (i-1)] );
		denominador = ( mat_perm[tam_mat_real + i] + mat_perm[tam_mat_real + (i-1)] );
		keff = numerador / denominador;
		b_L[tam_mat_real + i] = C*h/keff;
		
		numerador = ( 2 * mat_perm[tam_mat_real + i] * mat_perm[tam_mat_real + (i+1)] );
		denominador = ( mat_perm[tam_mat_real + i] + mat_perm[tam_mat_real + (i+1)] );
		keff = numerador / denominador;
		b_R[tam_mat_real + i] = C*h/keff;
		
		numerador = ( 2 * mat_perm[tam_mat_real + i] * mat_perm[(2 * tam_mat_real) + i] );
		denominador = ( mat_perm[tam_mat_real + i] + mat_perm[(2 * tam_mat_real) + i] );
		keff = numerador / denominador;
		b_D[tam_mat_real + i] = C*h/keff;
		
		
		//Calcula fronteira esquerda
		numerador = ( 2 * mat_perm[(i * tam_mat_real) + 1] * mat_perm[((i - 1) * tam_mat_real) + 1] );
		denominador = ( mat_perm[(i * tam_mat_real) + 1] + mat_perm[((i - 1) * tam_mat_real) + 1] );
		keff = numerador / denominador;
		b_U[(i * tam_mat_real) + 1] = C*h/keff;
		
		numerador = ( 2 * mat_perm[(i * tam_mat_real) + 1] * mat_perm[(i * tam_mat_real) + 2] );
		denominador = ( mat_perm[(i * tam_mat_real) + 1] + mat_perm[(i * tam_mat_real) + 2] );
		keff = numerador / denominador;
		b_R[(i * tam_mat_real) + 1] = C*h/keff;
		
		numerador = ( 2 * mat_perm[(i * tam_mat_real) + 1] * mat_perm[((i + 1) * tam_mat_real) + 1] );
		denominador = ( mat_perm[(i * tam_mat_real) + 1] + mat_perm[((i + 1) * tam_mat_real) + 1] );
		keff = numerador / denominador;
		b_D[(i * tam_mat_real) + 1] = C*h/keff;
		
		//Calcula fronteira inferior
		numerador = ( 2 * mat_perm[(tam_mat_interna * tam_mat_real) + i] * mat_perm[(tam_mat_interna * tam_mat_real) + (i - 1)] );
		denominador = ( mat_perm[(tam_mat_interna * tam_mat_real) + i] + mat_perm[(tam_mat_interna * tam_mat_real) + (i - 1)] );
		keff = numerador / denominador;
		b_L[(tam_mat_interna * tam_mat_real) + i] = C*h/keff;
		
		numerador = ( 2 * mat_perm[(tam_mat_interna * tam_mat_real) + i] * mat_perm[((tam_mat_interna - 1) * tam_mat_real) + i] );
		denominador = ( mat_perm[(tam_mat_interna * tam_mat_real) + i] + mat_perm[((tam_mat_interna - 1) * tam_mat_real) + i] );
		keff = numerador / denominador;
		b_U[(tam_mat_interna * tam_mat_real) + i] = C*h/keff;
		
		numerador = ( 2 * mat_perm[(tam_mat_interna * tam_mat_real) + i] * mat_perm[(tam_mat_interna * tam_mat_real) + (i + 1)] );
		denominador = ( mat_perm[(tam_mat_interna * tam_mat_real) + i] + mat_perm[(tam_mat_interna * tam_mat_real) + (i + 1)] );
		keff = numerador / denominador;
		b_R[(tam_mat_interna * tam_mat_real) + i] = C*h/keff;
		
		//Calcula fronteira direita
		numerador = ( 2 * mat_perm[(i * tam_mat_real) + tam_mat_interna] * mat_perm[((i-1) * tam_mat_real) + tam_mat_interna] );
		denominador = ( mat_perm[(i * tam_mat_real) + tam_mat_interna] + mat_perm[((i-1) * tam_mat_real) + tam_mat_interna] );
		keff = numerador / denominador;
		b_U[(i * tam_mat_real) + tam_mat_interna] = C*h/keff;
		
		numerador = ( 2 * mat_perm[(i * tam_mat_real) + tam_mat_interna] * mat_perm[(i * tam_mat_real) + (tam_mat_interna - 1)] );
		denominador = ( mat_perm[(i * tam_mat_real) + tam_mat_interna] + mat_perm[(i * tam_mat_real) + (tam_mat_interna - 1)] );
		keff = numerador / denominador;
		b_L[(i * tam_mat_real) + tam_mat_interna] = C*h/keff;
		
		numerador = ( 2 * mat_perm[(i * tam_mat_real) + tam_mat_interna] * mat_perm[((i+1) * tam_mat_real) + tam_mat_interna] );
		denominador = ( mat_perm[(i * tam_mat_real) + tam_mat_interna] + mat_perm[((i+1) * tam_mat_real) + tam_mat_interna] );
		keff = numerador / denominador;
		b_D[(i * tam_mat_real) + tam_mat_interna] = C*h/keff;
		
		//Calcula dados internos
		int j = 0;
		for(j = 2; j < tam_mat_interna; j ++){
			numerador = ( 2 * mat_perm[(i * tam_mat_real) + j] * mat_perm[(i * tam_mat_real) + (j - 1)] );
			denominador = ( mat_perm[(i * tam_mat_real) + j] + mat_perm[(i * tam_mat_real) + (j - 1)] );
			keff = numerador / denominador;
			b_L[(i * tam_mat_real) + j] = C*h/keff;
		
			numerador = ( 2 * mat_perm[(i * tam_mat_real) + j] * mat_perm[(i * tam_mat_real) + (j + 1)] );
			denominador = ( mat_perm[(i * tam_mat_real) + j] + mat_perm[(i * tam_mat_real) + (j + 1)] );
			keff = numerador / denominador;
			b_R[(i * tam_mat_real) + j] = C*h/keff;
		
			numerador = ( 2 * mat_perm[(i * tam_mat_real) + j] * mat_perm[((i - 1) * tam_mat_real) + j] );
			denominador = ( mat_perm[(i * tam_mat_real) + j] + mat_perm[((i - 1) * tam_mat_real) + j] );
			keff = numerador / denominador;
			b_U[(i * tam_mat_real) + j] = C*h/keff;
		
			numerador = ( 2 * mat_perm[(i * tam_mat_real) + j] * mat_perm[((i + 1) * tam_mat_real) + j] );
			denominador = ( mat_perm[(i * tam_mat_real) + j] + mat_perm[((i + 1) * tam_mat_real) + j] );
			keff = numerador / denominador;
			b_D[(i * tam_mat_real) + j] = C*h/keff;
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
	double constante = 2/h;
	
	for(i = 0; i < tam_mat_real; i ++)
		for(j = 0; j < tam_mat_real; j++){
			mat_epsilon[i*tam_mat_real + j] = constante * mat_perm[i*tam_mat_real + j];
			mat_font[i*tam_mat_real + j] *= h;
		}
	
	return 0;
}


char copia_dados_para_gpu(){
	
	HANDLE_ERROR( cudaMemcpy( dev_q_R, q_R, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_q_L, q_L, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_q_U, q_U, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_q_D, q_D, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	
	HANDLE_ERROR( cudaMemcpy( dev_q_R_old, q_R_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_q_L_old, q_L_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_q_U_old, q_U_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_q_D_old, q_D_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	
	
	HANDLE_ERROR( cudaMemcpy( dev_l_R, l_R, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_l_L, l_L, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_l_U, l_U, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_l_D, l_D, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	
	HANDLE_ERROR( cudaMemcpy( dev_l_R_old, l_R_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_l_L_old, l_L_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_l_U_old, l_U_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_l_D_old, l_D_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	
	
	HANDLE_ERROR( cudaMemcpy( dev_b_R, b_R, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b_L, b_L, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b_U, b_U, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b_D, b_D, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	
	HANDLE_ERROR( cudaMemcpy( dev_b_R_old, b_R_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b_L_old, b_L_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b_U_old, b_U_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b_D_old, b_D_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	
	HANDLE_ERROR( cudaMemcpy( dev_pressao, pressao, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_pressao_old, pressao_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	
	HANDLE_ERROR( cudaMemcpy( dev_mat_font, mat_font, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_mat_perm, mat_perm, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_mat_epsilon, mat_epsilon, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	return 0;
}

void copia_dados_para_cpu(){
	
	HANDLE_ERROR( cudaMemcpy( q_R, dev_q_R, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( q_L, dev_q_L, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( q_U, dev_q_U, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( q_D, dev_q_D, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	
	HANDLE_ERROR( cudaMemcpy( q_R_old, dev_q_R_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( q_L_old, dev_q_L_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( q_U_old, dev_q_U_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( q_D_old, dev_q_D_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	
	
	HANDLE_ERROR( cudaMemcpy( l_R, dev_l_R, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( l_L, dev_l_L, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( l_U, dev_l_U, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( l_D, dev_l_D, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	
	HANDLE_ERROR( cudaMemcpy( l_R_old, dev_l_R_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( l_L_old, dev_l_L_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( l_U_old, dev_l_U_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( l_D_old, dev_l_D_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	
	
	HANDLE_ERROR( cudaMemcpy( b_R, dev_b_R, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( b_L, dev_b_L, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( b_U, dev_b_U, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( b_D, dev_b_D, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	
	HANDLE_ERROR( cudaMemcpy( b_R_old, dev_b_R_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( b_L_old, dev_b_L_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( b_U_old, dev_b_U_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( b_D_old, dev_b_D_old, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyDeviceToHost ) );
	
	HANDLE_ERROR( cudaMemcpy( mat_font, dev_mat_font, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( mat_perm, dev_mat_perm, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( mat_epsilon, dev_mat_epsilon, tam_mat_real * tam_mat_real * sizeof(double),
                              cudaMemcpyHostToDevice ) );
}