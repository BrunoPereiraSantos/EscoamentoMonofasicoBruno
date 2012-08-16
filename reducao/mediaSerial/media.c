#include <stdio.h>
#include <stdlib.h>
#define N 100

int main (void){
	FILE *aqv = NULL;
	
	aqv = fopen("entrada.txt", "r");
	if(aqv == NULL)
		printf("\nErro ao abrir aqv\n");

	float f = 0.0, soma = 0.0;
	int i;
	for(i = 0; i < N; i++){
		fscanf (aqv, "%f", &f);
		soma += f;
	}
	printf("\n\nvalor da soma = %f", soma);
	printf("\nvalor da media = %f\n", soma/64);
	fclose (aqv);
	return 0;
}