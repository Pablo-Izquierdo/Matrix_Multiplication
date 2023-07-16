#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define RAND rand() % 2500 + 1000

void write_matrix(FILE *file, int rows, int columns) {
    int i, j;
    int zero = 0;

    fprintf(file, "%d %d\n", rows, columns);

    for (i = 0; i < rows; i++) {
        for (j = 0; j < columns; j++) {
            if (j < i)
                fprintf(file, "%d ", zero);
            else
//              fprintf(file, "%d ", j+1);
                fprintf(file, "%d ", (RAND));
        }
        fprintf(file, "\n");
    }
}

int main(int argc, char **argv) {
    srand(time(NULL));
    int rows, columns;
    FILE *file;

    if ((file = fopen("matrix.in", "w")) == NULL) {
        perror("File INPUT!");
        exit(EXIT_FAILURE);
    }

/*
 * Por defecto genera matrices de 5x5
 */
    rows = 5;
    columns = 5;

    if (argc > 2)
    {
/*
 * En este caso se genera una matriz del tamaño indicado en la línea de comandos.
 */
        rows = atoi (argv[1]);    //RAND;
        columns = atoi (argv[2]); //RAND;
    }

    printf("Generating A: %dx%d\n", rows, columns);
    write_matrix(file, rows, columns);

//  rows = columns;
//  columns = 2000;//RAND;

    printf("Generating B: %dx%d\n", rows, columns);
    write_matrix(file, rows, columns);

    fclose(file);

    return EXIT_SUCCESS;
}

















