# include <getopt.h>
# include <ctype.h>
# include <stdlib.h>
# include <stdio.h>
# include <unistd.h>
# include <string.h>
# include <pmmintrin.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void suma2D_CPU(float* A, float* B, int N, int V);

void getParams (int argc, char** argv, char* nValue, char* bValue, char* vValue);

int isInteger (char* input);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void suma2D_SHMEM (float* A, float* B, int N, int V) {
    
    int offset, neighbour, mid_row, neigh_row, center_neigh;

    int local_i = threadIdx.x;
    int local_j = threadIdx.y;
    int local_id = local_i + local_j * blockDim.y;

    int global_i = blockDim.x * blockIdx.x + local_i;
    int global_j = blockDim.y * blockIdx.y + local_j;
    int global_id = global_i + global_j * N;

    // Se declara un arreglo en memoria compartida para cada bloque con
    // el maximo de hebras posibles para un bloque en CUDA.
    __shared__ float temp[1024];

    B[global_id] = 0.0;
    temp[local_id] = 0.0;
    
    for (offset = -V * (1 + N); offset <= V * (1 + N); offset++) {
        neighbour = global_id + offset;
        neigh_row = neighbour / N;
        mid_row = global_id / N;
        
        // Condicion para no considerar vecinos fuera de los limites de la imagen
        if ( (neighbour >= 0) && (neighbour < (N * N)) ) {
            center_neigh = global_id - (mid_row - neigh_row) * N;

            // Condicion para no considerar vecinos fuera de la vecindad
            if ( (neighbour >= (center_neigh - V)) && (neighbour <= (center_neigh + V)) ) {
                temp[local_id] = temp[local_id] + A[neighbour];
            }
        }
    }

    B[global_id] = temp[local_id];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__host__ int main(int argc, char** argv) {
    
    char* nValue = (char*)malloc(sizeof(char));
    char* bValue = (char*)malloc(sizeof(char)); 
    char* vValue = (char*)malloc(sizeof(char));
    getParams (argc, argv, nValue, bValue, vValue);

    int N = atoi(nValue);
    int Bs = atoi(bValue);
    int V = atoi(vValue);

    dim3 gridSize = dim3(N / Bs, N / Bs);
    dim3 blockSize = dim3(Bs, Bs);
    
    float* h_a = (float*)malloc( (N * N) * sizeof(float));
    float* h_b = (float*)malloc( (N * N) * sizeof(float));

    float* d_a;
    float* d_b;

    cudaMalloc((void**) &d_a, (N * N) * sizeof(float));
    cudaMalloc((void**) &d_b, (N * N) * sizeof(float));

    // SE LLENA LA IMAGEN CON VALORES ALEATORIOS
    for (int index = 0; index < (N * N); index++) {
        h_a[index] = (float) rand() / RAND_MAX; 
        printf("%f ", h_a[index]);

        if ( (index + 1) % N == 0)
            printf("\n");
    }

    printf("\n");
    
    cudaMemcpy(d_a, h_a, (N * N) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, (N * N) * sizeof(float), cudaMemcpyHostToDevice);

    suma2D_SHMEM<<<gridSize, blockSize>>>(d_a, d_b, N, V);

    cudaMemcpy(h_b, d_b, (N * N) * sizeof(float), cudaMemcpyDeviceToHost);

    for (int index = 0; index < (N * N); index++) {
        printf("%f ", h_b[index]);

        if ( (index + 1) % N == 0)
            printf("\n");
    }

    // Liberacion de memoria para el host y el device.
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);

    // Liberacion de memoria para la recepcion de parametros de entrada.
    free(nValue);
    free(bValue);
    free(vValue);

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void suma2D_CPU(float* A, float* B, int N, int V) {

    
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// - INPUTS: - argc: Largo del arreglo de argumentos argv.
//           - argv: Arreglo con los argumentos de entrada incluyendo en nombre del archivo.
//           - iValue: Nombre del archivo de entrada que contiene el archivo en formato binario (RAW).
//           - oValue: Nombre del archivo de salida con las secuencia ordenada en formato binario (RAW).
//           - nValue: Largo de la secuencia contenida en el archivo de entrada (Numero entero multiplo de 16).
//           - dValue: Bandera que controla el debug para imprimir los resultados por consola (1) o no (0).
// - OUTPUTS: -
// - DESCRIPTION: Procedimiento que obtiene los parametros entregados por consola y almacenados en la variable "argv", y los deposita en las variables
//                iValue, oValue, nValue y dValue, en cada caso verificando la validez del valor entragado para cada bandera. Si alguna de estas banderas
//                no cumple con los formatos especificados el programa es interrumpido.

void getParams (int argc, char** argv, char* nValue, char* bValue, char* vValue) {

    int c;
    while ( (c = getopt (argc, argv, "N:B:V:")) != -1) {

        switch (c) {
            case 'N':
                strcpy(nValue, optarg);
                if (!isInteger(nValue)) {
                    printf ("%s\n", "-------------------------------------------------------------------------");
                    printf (" => El argumento de -%c debe ser un ENTERO POSITIVO.\n", c);
                    printf (" => Programa abortado\n");
                    printf ("%s\n", "-------------------------------------------------------------------------");
                    exit(EXIT_FAILURE);
                }

                break;

            case 'B':
                strcpy(bValue, optarg);
                if (!isInteger(bValue)) {
                    printf ("%s\n", "-------------------------------------------------------------------------");
                    printf (" => El argumento de -%c debe ser un ENTERO POSITIVO.\n", c);
                    printf (" => Programa abortado\n");
                    printf ("%s\n", "-------------------------------------------------------------------------");
                    exit(EXIT_FAILURE);
                }

                break;
            
            case 'V':
                strcpy(vValue, optarg);
                if (!isInteger(vValue)) {
                    printf ("%s\n", "-------------------------------------------------------------------------");
                    printf (" => El argumento de -%c debe ser un ENTERO POSITIVO.\n", c);
                    printf (" => Programa abortado\n");
                    printf ("%s\n", "-------------------------------------------------------------------------");
                    exit(EXIT_FAILURE);
                }

                break;

            case '?':
                if ( (optopt == 'N') || (optopt == 'B') || (optopt == 'V') ) { 
                    printf ("%s\n", "-------------------------------------------------------------------------");
                    printf (" => La opcion -%c requiere un argumento.\n", optopt);
                    printf (" => Programa abortado\n");
                    printf ("%s\n", "-------------------------------------------------------------------------");
                    exit(EXIT_FAILURE);
                }

                else if (isprint (optopt)) {
                    printf ("%s\n", "-------------------------------------------------------------------------");
                    printf (" => Opcion -%c desconocida.\n", optopt);
                    printf (" => Programa abortado\n");
                    printf ("%s\n", "-------------------------------------------------------------------------");
                    exit(EXIT_FAILURE);
                }

            default:
                break;
            }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// - INPUTS: - input: Cadena de caracteres a evaluar si corresponde a un numero entero positivo o no
// - OUTPUTS: Valor booleano 1 si es entero positivo, 0 en caso contrario
// - DESCRIPTION: Verifica si una cadena de caracteres de entrada posee en cada una de sus posiciones un caracter que es
//                digito y es positivo

int isInteger (char* input) {

    int c;
    // Recorrer el argumento entragado en cadena de caracteres, verificando que cada uno de estos corresponde a un numero.
    for (c = 0; c < strlen(input); c++) {

        // Si no se cumple para alguno de los caracteres, significa que el argumento no corresponde a un entero positivo y retorna 0.
        if (!isdigit(input[c]))
            return 0;
    }
    return 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////// END ////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////