/**********************************************************************

	Posgraduacao Modelagem Computacional
	IPRJ-UERJ
	Disciplina: Computacao Paralela
	Proffesor: Luis Felipe Pereira Martins
	
	Programa para resolver la ecuacao de Poisson usando el metodo de
	descomposicao de dominio e algoritmo serial para permeabilidad 
	variavel.
	
	Autor: Dany Sanchez Dominguez
	Modificado: 12-04-2003
	Compilador gcc
	
***********************************************************************/

#include <stdio.h>
#include <math.h>
#include<time.h>

/*
 *	Numero de celulas da malha N = 2^n + 2
 */
#define N 10

#define filename "perm1.dat"
#define PamM 2e-11
#define S 0.5

/*
 * Fluxos atuales e antigos em cada um dos lados da celula espacial
 * U (Uper), D (Down), L (Left), R (Right)
 */
float qu[N][N],
       qu_old[N][N],
       qd[N][N],
       qd_old[N][N],
       qr[N][N],
       qr_old[N][N],
       ql[N][N],
       ql_old[N][N];

/*
 *  Multiplicadores de Lagrange em cada um dos lados da celula espacial
 */        
float lu[N][N],
       lu_old[N][N],
       ld[N][N],
       ld_old[N][N],
       lr[N][N],
       lr_old[N][N],
       ll[N][N],
       ll_old[N][N];
       
/*
 *	Pressoes atuais e antigas em cada uma das celulas
 */ 
float p[N][N],
       p_old[N][N];

/*
 *	Betas da condicao de Robin em cada um dos lados da celula espacial
 */        
float betau[N][N],
       betad[N][N],
       betar[N][N],
       betal[N][N];

       
float f[N][N],      /* fonte */
       perm[N][N],   /* permeabilidad das rochas */
       shi[N][N];	 /* variavel auxiliar valor de shi */

float size=25600.00; /* dimensao da regiao */


int main()
  {
    int j,
        k,   
        i,  /* variavels auxiliares na implementacao */
        n;  /* quatidade real de celulas espaciais n = N-2 */

     float aux, h, 
            AuxU, AuxD, AuxR, AuxL, /* Auxiliares para cada lado das celulas */
            DU, DD, DR, DL,         /* Auxiliares para cada lado das celulas */
            M, erro,                /* Media das pressoes e erro na norma */
            sum1, sum2,             /* Auxiliares somadores */
            c = 1.,                 /* Valor para o calculo de beta */
            Keff;                   /* Valor medio da permeabilidade */
            
     float t;  
     
     FILE *arq;     
            
   clock_t ticks1, ticks2;  
   ticks1 = clock();        

  /*
   * Inicializacao das variaveis
   */
    for(j=1; j<N-1; j++)
      for(k=1; k<N-1; k++)
	      {
           f[j][k]=0.0;
           lu_old[j][k] = ld_old[j][k] = lr_old[j][k] = ll_old[j][k] = 0.0;
           qu_old[j][k] = qd_old[j][k] = qr_old[j][k] = ql_old[j][k]=0.0;
           p_old[j][k]=0.0;           
	      }
    
	/*
	 * Calculo de alguns auxiliares asignando valor de permeabilidade
	 * en blocos verticais
	 */      	  
	  n = N-2;    
	  h = size/n;	  

		
		arq = fopen(filename, "r");
		if (arq==NULL) return 0;		 
				
		for (k=n; k>0; k--)
			{
				fscanf(arq, "%d", &i);				
				if (i!=(n-k)) 
					{
						printf("Erro na linha %d.\n", n-k);
						return 0;
					}
				for (j=1; j<=n; j++)
					{
						fscanf(arq,"%g", &t);
						perm[j][k] = PamM*exp(S*t);						
				  }	
				fscanf(arq, "%d \n", &i);
				if (i!=192837465) 
					{
						printf("Erro na linha %d.\n", n-k);
						return 0;
					}				
			}		 
		 
		fclose(arq); 	  
	  
	  
/*	  i = n/2;    
	  for(j=1; j<=i; j++)
      for(k=1; k<N-1; k++)
	      {     
	        perm[j][k] = 1.0e-10;    
	        perm[j+i][k] = 1.0e-11; 
        }     */

    printf("permeabilidad \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf("%12.4E ",perm[j][k]);
	      printf("\n");
      }  
        
    /*
     * Calcula os Beta da Condicao de Robin
     */ 

    Keff = (2*perm[1][1]*perm[1][2])/(perm[1][2]+perm[1][1]);
    betau[1][1] = c*h/Keff;
    Keff = (2*perm[1][1]*perm[2][1])/(perm[1][1]+perm[2][1]);
    betar[1][1] = c*h/Keff;
    
    Keff = (2*perm[1][n]*perm[1][n-1])/(perm[1][n]+perm[1][n-1]);
    betad[1][n] = c*h/Keff;
    Keff = (2*perm[1][n]*perm[2][n])/(perm[1][n]+perm[2][n]);
    betar[1][n] = c*h/Keff;
    
    Keff = (2*perm[n][1]*perm[n][2])/(perm[n][1]+perm[n][2]);
    betau[n][1] = c*h/Keff;
    Keff = (2*perm[n][1]*perm[n-1][1])/(perm[n][1]+perm[n-1][1]);
    betal[n][1] = c*h/Keff;
    
    Keff = (2*perm[n][n]*perm[n][n-1])/(perm[n][n]+perm[n][n-1]);
    betad[n][n] = c*h/Keff;
    Keff = (2*perm[n][n]*perm[n-1][n])/(perm[n][n]+perm[n-1][n]);
    betal[n][n] = c*h/Keff;
    
    for (i=2; i<n; i++)
    	{
	    	Keff = (2*perm[i][1]*perm[i][2])/(perm[i][1]+perm[i][2]);
    		betau[i][1] = c*h/Keff;
    		Keff = (2*perm[i][1]*perm[i-1][1])/(perm[i][1]+perm[i-1][1]);
    		betal[i][1] = c*h/Keff;
    		Keff = (2*perm[i][1]*perm[i+1][1])/(perm[i][1]+perm[i+1][1]);
    		betar[i][1] = c*h/Keff;
    			
    		Keff = (2*perm[i][n]*perm[i][n-1])/(perm[i][n]+perm[i][n-1]);
    		betad[i][n] = c*h/Keff;
    		Keff = (2*perm[i][n]*perm[i-1][n])/(perm[i][n]+perm[i-1][n]);
    		betal[i][n] = c*h/Keff;
    		Keff = (2*perm[i][n]*perm[i+1][n])/(perm[i][n]+perm[i+1][n]);
    		betar[i][n] = c*h/Keff;
    		
    		Keff = (2*perm[1][i]*perm[1][i+1])/(perm[1][i]+perm[1][i+1]);
    		betau[1][i] = c*h/Keff;
    		Keff = (2*perm[1][i]*perm[1][i-1])/(perm[1][i]+perm[1][i-1]);
    		betad[1][i] = c*h/Keff;
    		Keff = (2*perm[1][i]*perm[2][i])/(perm[1][i]+perm[2][i]);
    		betar[1][i] = c*h/Keff;
    		
				Keff = (2*perm[n][i]*perm[n][i+1])/(perm[n][i]+perm[n][i+1]);
    		betau[n][i] = c*h/Keff;
    		Keff = (2*perm[n][i]*perm[n][i-1])/(perm[n][i]+perm[n][i-1]);
    		betad[n][i] = c*h/Keff;
    		Keff = (2*perm[n][i]*perm[n-1][i])/(perm[n][i]+perm[n-1][i]);
    		betal[n][i] = c*h/Keff;   		
    	}
    
    for(j=2; j<n; j++)
      for(k=2; k<n; k++)
	      {     
		    Keff = (2*perm[j][k]*perm[j][k+1])/(perm[j][k]+perm[j][k+1]);
	        betau[j][k] = c*h/Keff;
	        Keff = (2*perm[j][k]*perm[j][k-1])/(perm[j][k]+perm[j][k-1]);
	        betad[j][k] = c*h/Keff;
	        Keff = (2*perm[j][k]*perm[j+1][k])/(perm[j][k]+perm[j+1][k]);
	        betar[j][k] = c*h/Keff;
	        Keff = (2*perm[j][k]*perm[j-1][k])/(perm[j][k]+perm[j-1][k]);
	        betal[j][k] = c*h/Keff;
          }     
    
	  printf("betau \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf("%12.4E ",betau[j][k]);
	      printf("\n");
      }

    printf("betar \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf("%12.4E ",betar[j][k]);
	      printf("\n");
      }

    printf("betal \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf("%12.4E ",betal[j][k]);
	      printf("\n");
      }

    printf("betad \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf("%12.4E ",betad[j][k]);
	      printf("\n");
      }		      
	
      
      
    printf("Salida del serial para comparar\n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf("\n j %d k %d   perm %12.4E  betal %12.4E betar %12.4E betau %12.4E betad %12.4E \n",j, k, perm[j][k], 
	  		          betal[j][k], betar[j][k], betau[j][k], betad[j][k]);
	      printf("\n");
      }		           
     //exit(1); 
     
      
    /*
     * Asignando os valores da fonte
     */  
    f[1][1]=1.0e-7;
    //f[n][n]=-1.0e-7;
	f[n][n]=0.0;

    //Impressao do termo fonte
    printf("Fonte \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf(" %6.4E ",f[j][k]);
	      printf("\n ");
      }

    printf("lu_old \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf(" %6.4E ",lu_old[j][k]);
	      printf("\n ");
      }

    printf("lr_old \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf(" %6.4E ",lr_old[j][k]);
	      printf("\n ");
      }

    printf("ll_old \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf(" %6.4E ",ll_old[j][k]);
	      printf("\n ");
      }

    printf("ld_old \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf(" %6.4E ",ld_old[j][k]);
	      printf("\n ");
      }

    printf("qu_old \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf(" %6.4E ",qu_old[j][k]);
	      printf("\n ");
      }

    printf("qr_old \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf(" %6.4E ",qr_old[j][k]);
	      printf("\n ");
      }

    printf("ql_old \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf(" %6.4E ",ql_old[j][k]);
	      printf("\n ");
      }

    printf("qd_old \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf(" %6.4E ",qd_old[j][k]);
	      printf("\n ");
      }

    printf("betau \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf(" %6.4E ",betau[j][k]);
	      printf("\n ");
      }

    printf("betar \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf(" %6.4E ",betar[j][k]);
	      printf("\n ");
      }

    printf("betal \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf(" %6.4E ",betal[j][k]);
	      printf("\n ");
      }

    printf("betad \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf(" %6.4E ",betad[j][k]);
	      printf("\n ");
      }

   

    
	/*
	 * calculo de parametros que nao dependem das iteracoes
	 */
    aux = 1/h;
    for (k=1; k<=n; k++)
    	for (j=1; j<=n; j++)
    	  shi[j][k]=2*perm[j][k]*aux;

   printf("shi \n");
    for(k=n ;k>0; k--)
      {
        for(j=1; j<N-1; j++)
	  		  printf(" %6.4E ",shi[j][k]);
	      printf("\n ");
      }

    /*
     * Ciclo infinito para as iteracoes
     */
  i = 0;     
	for (i = 0; i < 10 ;)
		{		    
			  /*Calculo da pressao e dos fluxos em cada elemento */

		    /*canto inferior izquierdo [1][1]*/
		    AuxU = shi[1][1]/(1+betau[1][1]*shi[1][1]);
		    AuxR = shi[1][1]/(1+betar[1][1]*shi[1][1]);
		    DU = AuxU*(betau[1][1]*qd_old[1][2]+ld_old[1][2]);
		    DR = AuxR*(betar[1][1]*ql_old[2][1]+ll_old[2][1]);
		    p[1][1] = (f[1][1] * h + DU + DR)/(AuxU + AuxR);
		    qu[1][1] = AuxU*p[1][1] - DU;
		    qr[1][1] = AuxR*p[1][1] - DR;

		    /*canto superior izquierdo [1][N-2]*/
		    AuxD = shi[1][n]/(1+betad[1][n]*shi[1][n]);
		    AuxR = shi[1][n]/(1+betar[1][n]*shi[1][n]);
		    DD = AuxD*(betad[1][n]*qu_old[1][n-1]+lu_old[1][n-1]);
		    DR = AuxR*(betar[1][n]*ql_old[2][n]+ll_old[2][n]);
		    p[1][n] = (f[1][n] * h + DD + DR)/(AuxD + AuxR);
		    qd[1][n] = AuxD * p[1][n] - DD;
		    qr[1][n] = AuxR * p[1][n] - DR;

		    /*canto inferior dereito [N-2][1]*/
		    AuxU = shi[n][1]/(1+betau[n][1]*shi[n][1]);
		    AuxL = shi[n][1]/(1+betal[n][1]*shi[n][1]);
		    DU = AuxU*(betau[n][1]*qd_old[n][2]+ld_old[n][2]);
		    DL = AuxL*(betal[n][1]*qr_old[n-1][1]+lr_old[n-1][1]);
		    p[n][1] = (f[n][1] * h + DU + DL)/(AuxU + AuxL);
		    qu[n][1] = AuxU*p[n][1] - DU;
		    ql[n][1] = AuxL*p[n][1] - DL;

		    /*canto superior dereito [N-2][N-2]*/
		    AuxD = shi[n][n]/(1+betad[n][n]*shi[n][n]);
		    AuxL = shi[n][n]/(1+betal[n][n]*shi[n][n]);
		    DD = AuxD*(betad[n][n]*qu_old[n][n-1]+lu_old[n][n-1]);
		    DL = AuxL*(betal[n][n]*qr_old[n-1][n]+lr_old[n-1][n]);
		    p[n][n] = (f[n][n] * h + DD + DL)/(AuxD + AuxL);
		    qd[n][n] = AuxD*p[n][n] - DD;
		    ql[n][n] = AuxL*p[n][n] - DL;

		    /*fronteira U [2...N-3][N-2]*/
		    for (j=2; j<n; j++)
		    	{
			    	AuxL = shi[j][n]/(1+betal[j][n]*shi[j][n]);
			    	AuxR = shi[j][n]/(1+betar[j][n]*shi[j][n]);
			    	AuxD = shi[j][n]/(1+betad[j][n]*shi[j][n]);
			    	DL = AuxL*(betal[j][n]*qr_old[j-1][n]+lr_old[j-1][n]);
			    	DR = AuxR*(betar[j][n]*ql_old[j+1][n]+ll_old[j+1][n]);
			    	DD = AuxD*(betad[j][n]*qu_old[j][n-1]+lu_old[j][n-1]);
			    	p[j][n] = (f[j][n] * h + DD + DL + DR)/(AuxD + AuxL + AuxR);
			    	ql[j][n] = AuxL*p[j][n] - DL;
			    	qr[j][n] = AuxR*p[j][n] - DR;
			    	qd[j][n] = AuxD*p[j][n] - DD;
		    	}

		    /*fronteira D [2...N-3][1]*/
			for (j=2; j<n; j++)
		    	{
			    	AuxL = shi[j][1]/(1+betal[j][1]*shi[j][1]);
			    	AuxR = shi[j][1]/(1+betar[j][1]*shi[j][1]);
			    	AuxU = shi[j][1]/(1+betau[j][1]*shi[j][1]);
			    	DL = AuxL*(betal[j][1]*qr_old[j-1][1]+lr_old[j-1][1]);
			    	DR = AuxR*(betar[j][1]*ql_old[j+1][1]+ll_old[j+1][1]);
			    	DU = AuxU*(betau[j][1]*qd_old[j][2]+ld_old[j][2]);
			    	p[j][1] = (f[j][1] * h + DU + DL + DR)/(AuxU + AuxL + AuxR);
			    	ql[j][1] = AuxL*p[j][1] - DL;
			    	qr[j][1] = AuxR*p[j][1] - DR;
			    	qu[j][1] = AuxU*p[j][1] - DU;
		    	}

		   /*fronteira R [N-2][2...N-3]*/
		   for (k=2; k<n; k++)
		    {
			   	AuxU = shi[n][k]/(1+betau[n][k]*shi[n][k]);
			    AuxD = shi[n][k]/(1+betad[n][k]*shi[n][k]);
			    AuxL = shi[n][k]/(1+betal[n][k]*shi[n][k]);
			    DU = AuxU*(betau[n][k]*qd_old[n][k+1]+ld_old[n][k+1]);
			    DD = AuxD*(betad[n][k]*qu_old[n][k-1]+lu_old[n][k-1]);
			    DL = AuxL*(betal[n][k]*qr_old[n-1][k]+lr_old[n-1][k]);
			    p[n][k] = (f[n][k] * h + DU + DL + DD)/(AuxU + AuxL + AuxD);
			    qu[n][k] = AuxU*p[n][k] - DU;
			    qd[n][k] = AuxD*p[n][k] - DD;
			    ql[n][k] = AuxL*p[n][k] - DL;
		    }

		   /*fronteira L [1][2...N-3]*/
		   for (k=2; k<n; k++)
		   	{
			   	AuxU = shi[1][k]/(1+betau[1][k]*shi[1][k]);
			   	AuxD = shi[1][k]/(1+betad[1][k]*shi[1][k]);
			   	AuxR = shi[1][k]/(1+betar[1][k]*shi[1][k]);
			   	DU = AuxU*(betau[1][k]*qd_old[1][k+1]+ld_old[1][k+1]);
			   	DD = AuxD*(betad[1][k]*qu_old[1][k-1]+lu_old[1][k-1]);
			   	DR = AuxR*(betar[1][k]*ql_old[2][k]+ll_old[2][k]);
			   	p[1][k] = (f[1][k] * h + DU + DR + DD)/(AuxU + AuxR + AuxD);
			   	qu[1][k] = AuxU*p[1][k] - DU;
			   	qd[1][k] = AuxD*p[1][k] - DD;
			   	qr[1][k] = AuxR*p[1][k] - DR;
		   	}

		   /*elementos interiores [2..N-3][2..N-3]*/
		   for (k=2; k<n; k++)
		     for (j=2; j<n; j++)
		     	{
			    	AuxL = shi[j][k]/(1+betal[j][k]*shi[j][k]);
			    	AuxR = shi[j][k]/(1+betar[j][k]*shi[j][k]);
			    	AuxU = shi[j][k]/(1+betau[j][k]*shi[j][k]);
			    	AuxD = shi[j][k]/(1+betad[j][k]*shi[j][k]);
			    	DL = AuxL*(betal[j][k]*qr_old[j-1][k]+lr_old[j-1][k]);
			    	DR = AuxR*(betar[j][k]*ql_old[j+1][k]+ll_old[j+1][k]);
			    	DU = AuxU*(betau[j][k]*qd_old[j][k+1]+ld_old[j][k+1]);
			    	DD = AuxD*(betad[j][k]*qu_old[j][k-1]+lu_old[j][k-1]);
			    	p[j][k] = (f[j][k] * h + DU + DD + DR + DL)/(AuxU + AuxR + AuxD + AuxL);
			    	ql[j][k] = AuxL*p[j][k] - DL;
			   	  qr[j][k] = AuxR*p[j][k] - DR;
			    	qu[j][k] = AuxU*p[j][k] - DU;
			   	  qd[j][k] = AuxD*p[j][k] - DD;
			    }

		  /*actualizando los multiplicadores de lagrange*/
		  for (k=1; k<=n; k++)
		     for (j=1; j<=n; j++)
		     	{
			     	lu[j][k] = betau[j][k]*(qu[j][k] + qd_old[j][k+1]) + ld_old[j][k+1];
			     	ld[j][k] = betad[j][k]*(qd[j][k] + qu_old[j][k-1]) + lu_old[j][k-1];
			     	lr[j][k] = betar[j][k]*(qr[j][k] + ql_old[j+1][k]) + ll_old[j+1][k];
			     	ll[j][k] = betal[j][k]*(ql[j][k] + qr_old[j-1][k]) + lr_old[j-1][k];
		     	}

		    //printf("##############################################################\n"); 	
		    printf("\n Presion sin media iter %d erro %6.4E\n", i, erro);
		    for(k=n ;k>0; k--)
		      {
		        for(j=1; j<N-1; j++)
			  		  printf("%11.3E",p[j][k]);
			      printf("\n");
		      }

        //Comprobacion de la simetria
        /*for (k=1; k<=n; k++)
          for (j=1; j<=n; j++)
            {
              if (p[j][k]!=-p[n-j+1][n-k+1])
                printf("\n\n DIFERENCIA %d %d %30.22E.\n\n", j, k, p[j][k]);
            }*/

		    /* 
		     * Imponiendo a media cero na distribuicao de presiones
		     * Calculo de la media
		     */
		    M = 0.0;
			for (k=1; k<=n; k++){
				for (j=1; j<=n; j++){
					M += p[j][k];
				}
			}
		    M = M / (n*n);	
			
			
			printf("\n\ni = %d - valor da media:%f\n\n", i, M);
		    /*Media zero nas pressoes e multiplicadores de lagrange*/
		    for (k=1; k<=n; k++)
		     for (j=1; j<=n; j++)
		    	{
			    	p[j][k] -= M;
			     	lu[j][k] -= M;
			     	ld[j][k] -= M;
			     	lr[j][k] -= M;
			     	ll[j][k] -= M;
		    	}

		    /*avaliando criterio de convergencia*/
		    sum1 = 0.;
		    sum2 = 0.;
		    for (k=1; k<=n; k++)
		     for (j=1; j<=n; j++)
		    	{
			    	aux = p[j][k] - p_old[j][k];
			    	sum1 += aux*aux;
			    	sum2 += p[j][k]*p[j][k];
		    	}
		    erro = sqrt(sum1/sum2);

		    //printf("##############################################################\n"); 	
		    printf("\n Presion iter %d erro %6.4E\n", i, erro);
		    for(k=n ;k>0; k--)
		      {
		        for(j=1; j<N-1; j++)
			  		  printf("%16.6E",p[j][k]);
			      printf("\n");
		      }
		        printf("qu \n");
				    for(k=n ;k>0; k--)
				      {
				        for(j=1; j<N-1; j++)
					  		  printf(" %6.4E ",qu[j][k]);
					      printf("\n ");
				      }

				    printf("qr \n");
				    for(k=n ;k>0; k--)
				      {
				        for(j=1; j<N-1; j++)
					  		  printf(" %6.4E ",qr[j][k]);
					      printf("\n ");
				      }

				    printf("ql \n");
				    for(k=n ;k>0; k--)
				      {
				        for(j=1; j<N-1; j++)
					  		  printf(" %6.4E ",ql[j][k]);
					      printf("\n ");
				      }

				    printf("qd \n");
				    for(k=n ;k>0; k--)
				      {
				        for(j=1; j<N-1; j++)
					  		  printf(" %6.4E ",qd[j][k]);
					      printf("\n ");
				      }
				      
				    printf("lu \n");
				    for(k=n ;k>0; k--)
				      {
				        for(j=1; j<N-1; j++)
					  		  printf(" %6.4E ",lu[j][k]);
					      printf("\n ");
				      }

				    printf("lr \n");
				    for(k=n ;k>0; k--)
				      {
				        for(j=1; j<N-1; j++)
					  		  printf(" %6.4E ",lr[j][k]);
					      printf("\n ");
				      }

				    printf("ll \n");
				    for(k=n ;k>0; k--)
				      {
				        for(j=1; j<N-1; j++)
					  		  printf(" %6.4E ",ll[j][k]);
					      printf("\n ");
				      }

				    printf("ld \n");
				    for(k=n ;k>0; k--)
				      {
				        for(j=1; j<N-1; j++)
					  		  printf(" %6.4E ",ld[j][k]);
					      printf("\n ");
				      }       
				      
				      printf("\n ");
		
			 
			 /*Criterio de convergencia satisfeito fim do calculo*/      
		   if (erro < 1e-5) break;
			
		   /*
		    * Criterio de convergencia nao satisfeito atualizacao das pressoes
		    * fluxos e multiplicadores antigos para uma nova iteracao	
		    */
			 for (k=1; k<=n; k++)
		     for (j=1; j<=n; j++)
		 			{
			 			p_old[j][k] = p[j][k];
			 			qu_old[j][k] = qu[j][k];
			 			qd_old[j][k] = qd[j][k];
			 			ql_old[j][k] = ql[j][k];
			 			qr_old[j][k] = qr[j][k];
						lu_old[j][k] = lu[j][k];
			 			ld_old[j][k] = ld[j][k];
			 			ll_old[j][k] = ll[j][k];
			 			lr_old[j][k] = lr[j][k];
		 			}
			i++;
    } /* fin do ciclo infinito for(;;) */
    
    
        printf("\n Presion iter %d erro %6.4E\n", i, erro);
		    for(k=n ;k>0; k--)
		      {
		        for(j=1; j<N-1; j++)
			  		  printf("%16.6E",p[j][k]);
			      printf("\n");
		    }  
    ticks2 = clock();
 	  printf("Time in seconds %ld, %ld, %g,  %g, %g", ticks2, ticks1, (ticks1+1.-1.)/CLOCKS_PER_SEC, (ticks2+1.-1.)/CLOCKS_PER_SEC, (ticks2+1.-1.)/CLOCKS_PER_SEC - (ticks1+1.-1.)/CLOCKS_PER_SEC);
    system("PAUSE");
    return 0; /* fim da funcao main */
  }

