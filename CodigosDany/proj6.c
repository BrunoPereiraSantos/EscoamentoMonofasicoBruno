#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define filename "perm.dat"    /* arquivo com campo fractal */
#define PamM 2e-11             /* parametro para calcular permeabilidade */
#define S 0.5                  /* parametro para calcular permeabilidade */

#define nsx 2     /* Numero de procesadores na direcao X */
#define nsy 4     /* Numero de procesadores na direcao Y */
#define n   128   /* Nro de celulas de reais do prblema */
#define NPx 66  	/* n/nsx+2, Nro de celulas em cada procesador em j (X) */
#define npx 64    /* NPx-2, Nro de celulas reais em cada procesador em j */
#define NPy 34 	  /* n/nsy+2, Nro de celulas em cada procesador em k (Y) */
#define npy 32    /* NPy-2, Nro de celulas reais em cada procesador em k */

/*
 * Fluxos atuais e antigos em cada um dos lados da celula espacial
 * U (Uper), D (Down), L (Left), R (Right)
 */ 
double qu[NPx][NPy],
       qu_old[NPx][NPy],
       qd[NPx][NPy],
       qd_old[NPx][NPy],
       qr[NPx][NPy],
       qr_old[NPx][NPy],
       ql[NPx][NPy],
       ql_old[NPx][NPy];

/*
 *  Multiplicadores de Lagrange em cada um dos lados da celula espacial
 */        
double lu[NPx][NPy],
       lu_old[NPx][NPy],
       ld[NPx][NPy],
       ld_old[NPx][NPy],
       lr[NPx][NPy],
       lr_old[NPx][NPy],
       ll[NPx][NPy],
       ll_old[NPx][NPy];
       
/*
 *	Pressoes atuais e antigas em cada uma das celulas
 */ 
double p[NPx][NPy],
       p_old[NPx][NPy];

/*
 *	Betas da condicao de Robin em cada um dos lados da celula espacial
 */        
double betau[NPx][NPy],
       betad[NPx][NPy],
       betar[NPx][NPy],
       betal[NPx][NPy];
       
double f[NPx][NPy],      /* fonte */
       perm[NPx][NPy],   /* permeabilidad das rochas */
       shi[NPx][NPy];	 /* variavel auxiliar valor de shi */

double size=25600.00; /* dimensao da regiao */

double AuxU, AuxD, AuxR, AuxL, /* Auxiliares para cada lado das celulas */
       DU, DD, DR, DL;         /* Auxiliares para cada lado das celulas */
       
int block_ID[2];  /* vetor auxiliar para estabelecer tipo de bloco */
int block_type;		/* variavel que define tipo de bloco 0..8 */	        
       
int rank;  /* almacena os ranks do mpi */


/*
 *  Funcoes para estabelecer o tipo de bloco 0..8 em funcao
 *  do rank
 */
void set_subdomain_ID();
void set_subdomain_type();

/*
 *	Funcoes para estabelecer ler o arquivo de dados de entrada e calcular
 *  as permeabilidades em cada um dos tipos de blocos existentes 0..8
 */
void lee_data_0();
void lee_data_1();
void lee_data_2();
void lee_data_3();
void lee_data_5();
void lee_data_6();
void lee_data_7();
void lee_data_8();
       
/*
 *	Programa principal
 */
main(int argc, char *argv[])
	{
		int		MPI_size;
		MPI_Status main_status;	/* variaveis auxiliares do MPI */
		
    int j,
        k,  /* variaveis auxiliares na implementacao */ 
        i;  /* numero de iteracoes */

     double aux, h, 								/* auxiliar, dimensoes da celula espacial */
            M, erro,                /* Media das pressoes e erro na norma */
            sum1, sum2,             /* Auxiliares somadores */
            c = 1.,                 /* Valor para o calculo de beta */
            Keff;                   /* Valor medio da permeabilidade */
            
		float t; 
		FILE *arq; 
		
		char nameoutfile[20];       
		char auxStr[10];

  /*
 	 * Inicializa MPI
 	 */
	MPI_Init(&argc, &argv);

	/*
 	 * Testando o numero de procesos
 	 */
	MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);
	if (nsx*nsy != MPI_size) {
		MPI_Finalize();
		printf("Erro na declaracao da malha de procesadores.\n");
		return(1);
	}
	
	/*
 	 * Establece el rank de cada procesador.
 	 */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	/*
	 *	Estabelece o tipo de bloco em funcao de cada um dos ranks
	 */	
	/*printf("Antes subdomian %d\n", rank); */
	set_subdomain_ID();
	/*printf("Apos subdomian %d\n", rank);*/
	set_subdomain_type();		
	printf("rank %d  block_type %d\n", rank, block_type);
            
  /*
   * Inicializacao das variaveis
   */    
    /*for(j=0; j<NPro; j++)
      for(k=1; k<=n; k++)
	      {
           f[j][k]=0.0;
           lu_old[j][k] = ld_old[j][k] = lr_old[j][k] = ll_old[j][k] = 0.0;
           qu_old[j][k] = qd_old[j][k] = qr_old[j][k] = ql_old[j][k]=0.0;
           p_old[j][k]=0.0;           
	      }
    
	/*
	 *  Leitura do campo fractal correspondente a cada procesador,
	 *  calculo das permeabilidades perm = S * PerM * exp(fractal)
	 */  
	  h = size/n;
    if (block_type==0) lee_data_0();
    else if (block_type==2) lee_data_2();
    else if (block_type==6) lee_data_6(); 
    else if (block_type==8) lee_data_8();
    else if (block_type==1) lee_data_1();
    else if (block_type==7) lee_data_7(); 
    else if (block_type==3) lee_data_3();
    else lee_data_5();    
	 
    
    /*Salida de cada procesador para um arquivo independente */ 
    strcpy(nameoutfile, "rank");      
    itoa(rank, auxStr, 10);
    strcat(nameoutfile, auxStr);
    strcat(nameoutfile, ".out"); 
	  /*printf("Nome arquivo %s\n", nameoutfile);	      */

	  arq = fopen(nameoutfile, "w");
		if (arq==NULL) 
  		{
	  		printf("Nao foi possivel, abrir arquivo para saida rank %d", rank);
	  		return 0;
  		}    
     fprintf(arq, "Permeabilidade\n");
		 for(k=npy+1; k>=0; k--)
		   {
		     for(j=0; j<=npx+1; j++)
			  	 fprintf(arq, "%16.6E",perm[j][k]);
			   fprintf(arq, "\n");
		   }    		
  	fclose(arq);	
    
    /*
     *	Iniacializacao das funcoes em cada um dos procesadores
     */    
/*    if (rank==0)
    	{         	
    		f[1][1] = 1.0e-7;
    		p_canto_d_l = canto_d_l;
    		p_canto_u_l = canto_u_l;
    		p_canto_d_r = fronteira_d;
    		p_canto_u_r = fronteira_u;
    		p_fronteira_u = fronteira_u;
    		p_fronteira_d = fronteira_d;
    		p_fronteira_l = fronteira_l;
    		p_fronteira_r = internos;
    		p_internos = internos;	
   		}		
    else 	
    	{     
    		f[np][n]=-1.0e-7;
    		p_canto_d_l = fronteira_d;
    		p_canto_u_l = fronteira_u;
    		p_canto_d_r = canto_d_r;
    		p_canto_u_r = canto_u_r;
    		p_fronteira_u = fronteira_u;
    		p_fronteira_d = fronteira_d;
    		p_fronteira_l = internos;
    		p_fronteira_r = fronteira_r;
    		p_internos = internos;
   		}		*/
     	   

    /*
     * Calcula os Beta da Condicao de Robin
     */ 
/*		for (k=1; k<=npy; k++)
			for (j=1; j<=npx; j++)
				{
					Keff = (2*perm[j][k]*perm[j][k+1])/(perm[j][k]+perm[j][k+1]);
		    	betau[j][k] = c*h/Keff;
		    	Keff = (2*perm[j][k]*perm[j][k-1])/(perm[j][k]+perm[j][k-1]);
		    	betad[j][k] = c*h/Keff;
		    	Keff = (2*perm[j][k]*perm[j+1][k])/(perm[j+1][k]+perm[j][k]);
		    	betar[j][k] = c*h/Keff;
					Keff = (2*perm[j][k]*perm[j-1][k])/(perm[j-1][k]+perm[j][k]);
		    	betal[j][k] = c*h/Keff;
				}

    /*
     *	Finaliza o MPI, fim do programa
     */
		MPI_Finalize();
		exit(0);
	}
	
void set_subdomain_ID()
	{	 
	 block_ID[0] = rank % nsx;
	 block_ID[1] = rank / nsx;	
	}
	
void set_subdomain_type()
	{
		/* 4 cantos */
		if (block_ID[0] == 0 && block_ID[1] == 0)
		    block_type = 0;
		else if (block_ID[0] == nsx-1 && block_ID[1] == 0)
		         block_type = 2;
		else if (block_ID[0] == nsx-1 && block_ID[1] == nsy-1)
		         block_type = 8;
		else if (block_ID[0] == 0 && block_ID[1] == nsy-1)
		         block_type = 6;
		         
		/* 4 fronteiras L, R, U, D */
		else if (block_ID[1] == 0)
		         block_type = 1;         
		else if (block_ID[0] == nsx-1)
		         block_type = 5;
		else if (block_ID[1] == nsy-1)
		         block_type = 7;         
		else if (block_ID[0] == 0)
		         block_type = 3;
		         
		/* blocos interiores */
		else block_type = 4;
	}
	
void lee_data_0()
	{
		int  FinJ, FinK, inCtrl, j, k;			
		float t;
		FILE *arq;
		
		FinJ = npx + 1;		
		FinK = n - npy - 1;
		
	 	arq = fopen(filename, "r");
		if (arq==NULL) 
  		{
	  		printf("Nao foi possivel, abrir arquivo rank %d\n", rank);
	  		exit(0);
  		}
  		
		for (k=npy; k>=0; k--)
			{
				fscanf(arq, "%d", &inCtrl);
				/*printf("linha %d \n", inCtrl);					*/
				if (inCtrl!=(npy-k)) 
					{
						printf("Erro na linha  %d - %d.\n", inCtrl, npy-k);
						exit(0);
					}
				for (j=1; j<=FinJ; j++)
				  {
					 	fscanf(arq,"%g", &t);
						perm[j][k] = PamM*exp(S*t); 
					}	
				for (j=FinJ+1; j<=n; j++)
					fscanf(arq,"%g", &t);
				fscanf(arq, "%d", &inCtrl);				
				if (inCtrl!=192837465) 
					{
						printf("Erro na linha %d - %d.\n", inCtrl, npy-k);
						exit(0);
					}										
			}			
		fclose(arq);				
	}	

void lee_data_1()
	{
		int  IniJ, FinJ, FinK, inCtrl, j, k;			
		float t;
		FILE *arq;
		
		IniJ = rank * nsx;
		FinJ = IniJ + npx + 1;		
		FinK = n - npy - 1;
		
	 	arq = fopen(filename, "r");
		if (arq==NULL) 
  		{
	  		printf("Nao foi possivel, abrir arquivo rank %d\n", rank);
	  		exit(0);
  		}
  		
		for (k=npy; k>=0; k--)
			{
				fscanf(arq, "%d", &inCtrl);
				/*printf("linha %d \n", inCtrl);					*/
				if (inCtrl!=(npy-k)) 
					{
						printf("Erro na linha  %d - %d.\n", inCtrl, npy-k);
						exit(0);
					}
				for (j=1; j<IniJ; j++)	
  				fscanf(arq,"%g", &t);				
  								
				for (j=0; j<=(npx+1); j++)
				  {
					 	fscanf(arq,"%g", &t);
						perm[j][k] = PamM*exp(S*t); 
					}	
				for (j=FinJ+1; j<=n; j++)
					fscanf(arq,"%g", &t);
				fscanf(arq, "%d", &inCtrl);				
				if (inCtrl!=192837465) 
					{
						printf("Erro na linha %d - %d.\n", inCtrl, npy-k);
						exit(0);
					}										
			}			
		fclose(arq);				
	}	
	
void lee_data_2()
	{
		int  IniJ, FinK, inCtrl, j, k;			
		float t;
		FILE *arq;
		
		IniJ = n - npx - 1;		
		FinK = n - npy - 1;
		
	 	arq = fopen(filename, "r");
		if (arq==NULL) 
  		{
	  		printf("Nao foi possivel, abrir arquivo rank %d\n", rank);
	  		exit(0);
  		}
  		
		for (k=npy; k>=0; k--)
			{
				fscanf(arq, "%d", &inCtrl);
				/*printf("linha %d \n", inCtrl);					*/
				if (inCtrl!=(npy-k)) 
					{
						printf("Erro na linha  %d - %d.\n", inCtrl, npy-k);
						exit(0);
					}
				for (j=0; j<IniJ; j++)
				 	fscanf(arq,"%g", &t);
				for (j=IniJ; j<n; j++)
					{
						fscanf(arq,"%g", &t);
						perm[j-IniJ][k] = PamM*exp(S*t);
					}
				/*printf("Ultimo numero %12.4f\n", t);						*/
				fscanf(arq, "%d", &inCtrl);				
				if (inCtrl!=192837465) 
					{
						printf("Erro final da linha %d - %d rank %d.\n", inCtrl, npy-k, rank);
						exit(0);
					}										
			}			
		fclose(arq);				
	}		

void lee_data_3()
	{
		int  FinJ, IniK, FinK, inCtrl, j, k, aux;			
		float t;
		FILE *arq;
		char MyStr[1400];
		
		FinJ = npx + 1;	
		IniK = n - (rank/nsx)*npy + 1;	
		FinK = IniK - npy - 1;
		
	 	arq = fopen(filename, "r");
		if (arq==NULL) 
  		{
	  		printf("Nao foi possivel, abrir arquivo rank %d\n", rank);
	  		exit(0);
  		}
  		
  	inCtrl = 1400;
  	for (k=1; k<=(n-IniK); k++)
  	  fgets(MyStr, inCtrl, arq);
  	
  	aux = ((rank/nsx)+1)*npy;
  	/*printf("aux %d, rank %d \n", aux, rank);*/
		for (k=npy+1; k>=0; k--)
			{
				fscanf(arq, "%d", &inCtrl);
				/*printf("linha %d \n", inCtrl);					*/
				if (inCtrl!=(aux-k)) 
					{
						printf("Erro na linha  %d - %d rank %d.\n", inCtrl, aux-k, rank);
						exit(0);
					}
				for (j=1; j<=FinJ; j++)
				  {
					 	fscanf(arq,"%g", &t);
						perm[j][k] = PamM*exp(S*t); 
					}	
				for (j=FinJ+1; j<=n; j++)
					fscanf(arq,"%g", &t);
				fscanf(arq, "%d", &inCtrl);				
				if (inCtrl!=192837465) 
					{
						printf("Erro final da linha %d - %d rank %d.\n", inCtrl, aux-k, rank);
						exit(0);
					}										
			}			
		fclose(arq);				
	}	
	
void lee_data_5()
	{
		int  IniJ, IniK, FinK, inCtrl, j, k, aux;			
		float t;
		FILE *arq;
		char MyStr[1400];
		
		IniJ = n - npx;	
		IniK = n - (rank/nsx)*npy + 1;	
		FinK = IniK - npy - 1;
		
	 	arq = fopen(filename, "r");
		if (arq==NULL) 
  		{
	  		printf("Nao foi possivel, abrir arquivo rank %d\n", rank);
	  		exit(0);
  		}
  		
  	inCtrl = 1400;
  	for (k=1; k<=(n-IniK); k++)
  	  fgets(MyStr, inCtrl, arq);
  	
  	aux = ((rank/nsx)+1)*npy;  	
		for (k=npy+1; k>=0; k--)
			{
				fscanf(arq, "%d", &inCtrl);
				/*printf("linha %d \n", inCtrl);					*/
				if (inCtrl!=(aux-k)) 
					{
						printf("Erro na linha  %d - %d rank %d.\n", inCtrl, aux-k, rank);
						exit(0);
					}
				for (j=1; j<IniJ; j++)
					fscanf(arq,"%g", &t);	
				for (j=0; j<=npx; j++)
				  {
					 	fscanf(arq,"%g", &t);
						perm[j][k] = PamM*exp(S*t); 
					}					
				fscanf(arq, "%d", &inCtrl);				
				if (inCtrl!=192837465) 
					{
						printf("Erro final da linha %d - %d rank %d.\n", inCtrl, aux-k, rank);
						exit(0);
					}										
			}			
		fclose(arq);				
	}	
				
void lee_data_6()
	{
		int  FinJ, IniK, inCtrl, j, k;			
		float t;
		char MyStr[1400];
		FILE *arq;
		
		FinJ = npx + 1;		
		IniK = npy + 1;
		
	 	arq = fopen(filename, "r");
		if (arq==NULL) 
  		{
	  		printf("Nao foi possivel, abrir arquivo rank %d\n", rank);
	  		exit(0);
  		}
  		
  	inCtrl = 1400;
  	for (k=1; k<(n-npy); k++)
  	  fgets(MyStr, inCtrl, arq);
  		
		for (k=npy+1; k>0; k--)
			{
				fscanf(arq, "%d", &inCtrl);
				/*printf("linha %d \n", inCtrl);					*/
				if (inCtrl!=(n-k)) 
					{
						printf("Erro na linha  %d - %d rank %d.\n", inCtrl, n-k, rank);
						exit(0);
					}
				for (j=1; j<=FinJ; j++)
				  {
					 	fscanf(arq,"%g", &t);
						perm[j][k] = PamM*exp(S*t); 
					}	
				for (j=FinJ+1; j<=n; j++)
					fscanf(arq,"%g", &t);
				fscanf(arq, "%d", &inCtrl);				
				if (inCtrl!=192837465) 
					{
						printf("Erro na linha %d - %d rank %d.\n", inCtrl, n-k, rank);
						exit(0);
					}										
			}			
		fclose(arq);				
	}	
	
void lee_data_7()
	{
		int  IniJ, FinJ, IniK, inCtrl, j, k;			
		float t;
		FILE *arq;
		char MyStr[1400];
		
		IniJ = (rank-nsx) * nsx;
		FinJ = IniJ + npx + 1;		
		IniK = npy + 1;
		
	 	arq = fopen(filename, "r");
		if (arq==NULL) 
  		{
	  		printf("Nao foi possivel, abrir arquivo rank %d \n", rank);
	  		exit(0);
  		}
  	
  	inCtrl = 1400;
  	for (k=1; k<(n-npy); k++)
  	  fgets(MyStr, inCtrl, arq);			  	
  			
		for (k=(npy+1); k>0; k--)
			{
				fscanf(arq, "%d", &inCtrl);
				/*printf("linha %d \n", inCtrl);					*/
				if (inCtrl!=(n-k)) 
					{
						printf("Erro na linha  %d - %d.\n", inCtrl, n-k);
						exit(0);
					}
					
				for (j=1; j<IniJ; j++)	
  				fscanf(arq,"%g", &t);				
  								
				for (j=0; j<=(npx+1); j++)
				  {
					 	fscanf(arq,"%g", &t);
						perm[j][k] = PamM*exp(S*t); 
					}	
				for (j=FinJ+1; j<=n; j++)
					fscanf(arq,"%g", &t);
				fscanf(arq, "%d", &inCtrl);				
				if (inCtrl!=192837465) 
					{
						printf("Erro na linha %d - %d.\n", inCtrl, npy-k);
						exit(0);
					}										
			}			
		fclose(arq);						
	}
	
void lee_data_8()
	{
		int  IniJ, IniK, inCtrl, j, k;			
		float t;
		char MyStr[1400];
		FILE *arq;
		
		IniJ = n - npx - 1;		
		IniK = n - npy - 1;
		
	 	arq = fopen(filename, "r");
		if (arq==NULL) 
  		{
	  		printf("Nao foi possivel, abrir arquivo rank %d \n", rank);
	  		exit(0);
  		}
  		
  		inCtrl = 1400;
  		for (k=1; k<(n-npy); k++)
  		  fgets(MyStr, inCtrl, arq);
  		
		for (k=npy+1; k>0; k--)
			{
				fscanf(arq, "%d", &inCtrl);
				/*printf("linha %d \n", inCtrl);					*/
				if (inCtrl!=(n-k)) 
					{
						printf("Erro na linha  %d - %d rank %d.\n", inCtrl, n-k, rank);
						exit(0);
					}
					
				for (j=0; j<IniJ; j++)
					fscanf(arq,"%g", &t);	
					
				for (j=IniJ; j<n; j++)
				  {
					 	fscanf(arq,"%g", &t);
						perm[j-IniJ][k] = PamM*exp(S*t); 
					}	
				
				fscanf(arq, "%d", &inCtrl);				
				if (inCtrl!=192837465) 
					{
						printf("Erro na linha %d - %d rank %d.\n", inCtrl, n-k, rank);
						exit(0);
					}										
			}			
		fclose(arq);				
	}	
	
		     
/* 
int lee_arquivo (int IniJ, int FinJ, int IniK, int FinK)
	{
		int j, k, inCtrl;
		char MyStr[1400];
		FILE *arq;
		float t;
		
		printf("IniJ %d  FinJ %d IniK %d FinK %d\n", IniJ, FinJ, IniK, FinK);
		
		arq = fopen(filename, "r");
		if (arq==NULL) 
  		{
	  		printf("Nao foi possivel, abrir arquivo rank %d", rank);
	  		return 0;
  		}
		
		for (k=n; k>FinK; k--)
		  fgets(MyStr, 0, arq);
		  
		for (k=FinK; k>=IniK; k--)
			{				
				fscanf(arq, "%d", &inCtrl);					
				if (inCtrl!=(FinK-k)) 
					{
						printf("Erro na linha %d - %d.\n", inCtrl, FinK-k);
						return 0;
					}
				
				for (j=1; j<IniJ; j++)
				  fscanf(arq,"%g", &t);	
				  
				for (j=IniJ; j<=FinJ; j++)
			    {
						fscanf(arq,"%g", &t);
						perm[j-IniJ+1][k-FinK+n/nsy] = PamM*exp(S*t);
						printf("j %d k %d %d %d t %f perm %E \n", j, k, j-IniJ+1,  k-FinK+n/nsy, t, perm[j-IniJ+1][k-FinK+n/nsy]);
			    }
			      
			  for (j=FinJ+1; j<=n; j++)
				  fscanf(arq,"%g", &t);
				
				fscanf(arq, "%d", &inCtrl);				
				if (inCtrl!=192837465) 
					{
						printf("Erro na linha %d - %d.\n", inCtrl, FinK-k);
						return 0;
					}  	  				  
			}
		fclose(arq); 		
		  
		return 1; 
	} 		*/