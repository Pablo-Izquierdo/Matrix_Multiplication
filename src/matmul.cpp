#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/times.h>
#include <omp.h>

#define RAND rand() % 100
#define DEBUG


void init_mat_inf ();
void init_mat_sup ();
void matmul ();
void matmul_Parallel ();
void matmul_sup ();
void matmul_sup_Parallel();
void matmul_inf ();
void print_mat (float *M, int dim);
void write_matrix();

/* Usage: ./matmul <dim> [block_size]*/

int main (int argc, char* argv[])
{
    int block_size = 0;
    int dim;
    float *A = 0, *B = 0, *C = 0;
    double TinicioParallel;
    double TinicioSeq;
    double TtotalSeq;
    double TtotalParallel;
    FILE *file;



   dim = atoi (argv[1]);
   if (argc == 3) block_size = atoi (argv[2]);

    printf("blocksize = %d", block_size);
   //-----RESERVO ESPACIO DE MEMORIA-----

   A = malloc(sizeof(float)*dim*dim);
   B = malloc(sizeof(float)*dim*dim);
   C = malloc(sizeof(float)*dim*dim);


   //----GENERO LAS MATRICES----
   init_mat_sup(dim, A);
   init_mat_sup(dim, B);

   //--------------------PROCESAMIENTO MATRICES---------------------

   //-------------PARTE 1----------------
   //Parte 1 Parallel
    #ifdef DEBUG
        printf("\n-------Parte 1-------\n");
        printf("Inicio Timer Paralelo\n");
    #endif

    TinicioParallel = omp_get_wtime();

    matmul_Parallel(A, B, C, dim);

    TtotalParallel = omp_get_wtime() - TinicioParallel;

    //Genero un fichero con la solucion
    if ((file = fopen("Solucion1_Parallel.in", "w")) == NULL) {
        perror("File INPUT!");
        exit(EXIT_FAILURE);
    }

    write_matrix(file, C, dim);

    //Parte 1 Secuencial
    #ifdef DEBUG
        printf("Tiempo Paralelo = %f \n", TtotalParallel);
        printf("Inicio Timer Secuencial\n");
    #endif

    TinicioSeq = omp_get_wtime();

    matmul(A, B, C, dim);

    TtotalSeq = omp_get_wtime() - TinicioSeq;

    #ifdef DEBUG
        printf("Tiempo Secuencial = %f \n", TtotalSeq);
        printf("SpeedUp = %f \n", TtotalSeq/TtotalParallel);
    #endif

    //Genero un fichero con la solucion
    if ((file = fopen("Solucion1_Seq.in", "w")) == NULL) {
        perror("File INPUT!");
        exit(EXIT_FAILURE);
    }

    write_matrix(file, C, dim);

    //-------------PARTE 2----------------

    init_mat_sup(dim, A);
    init_mat_inf(dim, B);

                        //Parte 2 Secuencial
    #ifdef DEBUG
        printf("\n-------Parte 2-------\n");
        printf("Inicio Timer Secuencial\n");
    #endif

    TinicioSeq = omp_get_wtime();

    matmul_sup(A, B, C, dim);

    TtotalSeq = omp_get_wtime() - TinicioSeq;


    //Genero un fichero con la solucion
    if ((file = fopen("Solucion2_Seq.in", "w")) == NULL) {
        perror("File INPUT!");
        exit(EXIT_FAILURE);
    }

    write_matrix(file, C, dim);

                        //Parte 2 Parallel
    #ifdef DEBUG
        printf("Tiempo Secuencial = %f \n\n", TtotalSeq);
        printf("Paralelo, Schedule Static con tamaño bloque = %d\n", block_size);
        printf("Inicio Timer .... Operacion.... Finalizo Timer\n");
    #endif
                            // ---Static---
    omp_set_schedule(omp_sched_static, block_size);

    TinicioParallel = omp_get_wtime();

    matmul_sup_Parallel(A, B, C, dim);

    TtotalParallel = omp_get_wtime() - TinicioParallel;


    //Genero un fichero con la solucion
    if ((file = fopen("Solucion2_ParallelStatic.in", "w")) == NULL) {
        perror("File INPUT!");
        exit(EXIT_FAILURE);
    }

    write_matrix(file, C, dim);

                            //---Dynamic---
    #ifdef DEBUG
        printf("Tiempo Paralelo Static = %f \n", TtotalParallel);
        printf("SpeedUp Static = %f \n\n", TtotalSeq/TtotalParallel);
        printf("Paralelo, Schedule Dynamic con tamaño bloque = %d\n", block_size);
        printf("Inicio Timer .... Operacion.... Finalizo Timer\n");
    #endif

    omp_set_schedule(omp_sched_dynamic, block_size);

    TinicioParallel = omp_get_wtime();

    matmul_sup_Parallel(A, B, C, dim);

    TtotalParallel = omp_get_wtime() - TinicioParallel;


    //Genero un fichero con la solucion
    if ((file = fopen("Solucion2_ParallelDynamic.in", "w")) == NULL) {
        perror("File INPUT!");
        exit(EXIT_FAILURE);
    }

    write_matrix(file, C, dim);

                            //---Guided---
    #ifdef DEBUG
        printf("Tiempo Paralelo Dynamic = %f \n", TtotalParallel);
        printf("SpeedUp Dynamic = %f \n\n", TtotalSeq/TtotalParallel);
        printf("Inicio Timer .... Operacion.... Finalizo Timer\n");
        printf("Paralelo, Schedule Guided con tamaño bloque = %d\n", block_size);
    #endif

    omp_set_schedule(omp_sched_guided, block_size);

    TinicioParallel = omp_get_wtime();

    matmul_sup_Parallel(A, B, C, dim);

    TtotalParallel = omp_get_wtime() - TinicioParallel;


    //Genero un fichero con la solucion
    if ((file = fopen("Solucion2_ParallelGuided.in", "w")) == NULL) {
        perror("File INPUT!");
        exit(EXIT_FAILURE);
    }

    write_matrix(file, C, dim);

    #ifdef DEBUG
        printf("Tiempo Paralelo Guided = %f \n", TtotalParallel);
        printf("SpeedUp Guided = %f \n", TtotalSeq/TtotalParallel);
    #endif


    exit (0);
}

void matmul_Parallel (float *A, float *B, float *C, int dim)
{
    int i, j, k;

    #pragma omp parallel private(i, j, k) shared(A, B, C, dim)
    {
        #pragma omp for
        for (i=0; i < dim; i++)
            for (j=0; j < dim; j++)
                C[i*dim+j] = 0.0;

        #pragma omp for
        for (i=0; i < dim; i++)
            for (j=0; j < dim; j++)
                for (k=0; k < dim; k++)
                    C[i*dim+j] += A[i*dim+k] * B[j+k*dim];
    }
}

void matmul_sup_Parallel (float *A, float *B, float *C, int dim)
{
    int i, j, k;
    #pragma omp parallel private(i, j, k) shared(A, B, C, dim)
    {

        #pragma omp for schedule(runtime)
        for (i=0; i < dim; i++)
            for (j=0; j < dim; j++)
                C[i*dim+j] = 0.0;

        #pragma omp for schedule(runtime)
        for (i=0; i < (dim-1); i++)
            for (j=0; j < (dim-1); j++)
                for (k=(i+1); k < dim; k++)
                    C[i*dim+j] += A[i*dim+k] * B[j+k*dim];
    }
}


void print_mat (float *C, int dim){
    int i, j;

    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            printf("%f ", C[i*dim+j]);
        }
        printf("\n");
    }

}

void write_matrix(FILE *file, float *C, int dim) {
    int i, j;

    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            fprintf(file ,"%f ", C[i*dim+j]);
        }
        fprintf(file, "\n");
    }
}



void init_mat_inf (int dim, float *M)
{
    int i,j;
    //int m,n,k;

    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            if (j >= i)
                M[i*dim+j] = 0.0;
            else
                M[i*dim+j] = RAND;
        }
    }
}

void init_mat_sup (int dim, float *M)
{
    int i,j;
    //int m,n,k;

    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            if (j <= i)
                M[i*dim+j] = 0.0;
            else
                M[i*dim+j] = RAND;
        }
    }
}


void matmul (float *A, float *B, float *C, int dim)
{
    int i, j, k;

    for (i=0; i < dim; i++)
        for (j=0; j < dim; j++)
            C[i*dim+j] = 0.0;

    for (i=0; i < dim; i++)
        for (j=0; j < dim; j++)
            for (k=0; k < dim; k++)
                C[i*dim+j] += A[i*dim+k] * B[j+k*dim];
}

void matmul_sup (float *A, float *B, float *C, int dim)
{
    int i, j, k;

    for (i=0; i < dim; i++)
        for (j=0; j < dim; j++)
            C[i*dim+j] = 0.0;

    for (i=0; i < (dim-1); i++)
        for (j=0; j < (dim-1); j++)
            for (k=(i+1); k < dim; k++)
                C[i*dim+j] += A[i*dim+k] * B[j+k*dim];


}

void matmul_inf (float *A, float *B, float *C, int dim)
{
    int i, j, k;

    for (i=0; i < dim; i++)
        for (j=0; j < dim; j++)
            C[i*dim+j] = 0.0;

    for (i=1; i < dim; i++)
        for (j=1; j < dim; j++)
            for (k=0; k < i; k++)
                C[i*dim+j] += A[i*dim+k] * B[j+k*dim];
}

