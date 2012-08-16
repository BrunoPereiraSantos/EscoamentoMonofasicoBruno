#include <stdio.h>
#include <stdlib.h>
#include <string.h> 


int main(void){
	FILE *arq = NULL;
	int tam = 10, i, j, temp, comp;
	//arq = fopen("../dir_entrada/parametro_entrada.txt", "r");
	arq = fopen("entrada.txt", "r");
	if(arq == NULL){
		printf("Erro ao abrir aquivo: 'parametro_entrada.txt'\n\t\tCertifique-se que o arquivo exite.\n");
		exit(1);
	}
	float *mat;
	comp = tam * tam;
	mat = (float *) calloc(comp, sizeof(float));
	
	for(i = 1; i < comp; i ++)
		fscanf(arq, "%f", &mat[i]);
	
	float soma = 0.0;
	for(i = 0; i < comp; i++)
		soma += mat[i];
	
	temp = (tam - 2) * (tam - 2);
	float media = soma / temp;
	printf("\nvalor da media: %f\n\n", media);
	
	for(i = 0; i < tam; i ++){
		for(j = 0; j < tam; j++)
			printf("%6.4E ", mat[i*tam + j]);
		printf("\n");
	}
	
	return 0;
}