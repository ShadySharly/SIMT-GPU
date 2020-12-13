# include <getopt.h>
# include <ctype.h>
# include <stdlib.h>
# include <stdio.h>
# include <unistd.h>
# include <string.h>
# include <pmmintrin.h>
# include <time.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void suma2D_CPU(float* A, float* B, int N, int V);

void getParams (int argc, char** argv, char* nValue, char* bValue, char* vValue);

int isInteger (char* input);

float pixelSum (float* image, int N);

void printImage (float* image, int N);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void suma2D (float* A, float* B, int N, int V) {
    
    // Se definen variables para realizar la suma de la vecindad de un determinado pixel de la imagen
    int offset, neighbour, mid_row, neigh_row, center_neigh;

    // Se declara el identificador local de cada hebra en un boque, considerando su coordenada x e y.
    int local_i = threadIdx.x;
    int local_j = threadIdx.y;

    // Se declara el identificador global de cada hebra en un boque, considerando su coordenada x e y.
    int global_i = blockDim.x * blockIdx.x + local_i;
    int global_j = blockDim.y * blockIdx.y + local_j;

    // Se determina el identificador global de cada hebra, basandose en los identificadores globales en 
    // termino de sus coordenadas.
    int global_id = global_i + global_j * N;

    // Se inicializa un vaor inicial 0.0 para la imagen de salida, a la cual se iran sumando los vecinos,
    // asi tambien como el pixel central
    B[global_id] = 0.0;

    // Se recorren el arreglo desde el pixel con la primera coordenada, hasta la ultima coordenada, 
    // pasando por pixels que no son de la vecindad inclusive.
    for (offset = -V * (1 + N); offset <= V * (1 + N); offset++) {
        neighbour = global_id + offset; // Se determina la posicion del vecino del pixel central.
        neigh_row = neighbour / N;  // La fila del vecino
        mid_row = global_id / N;    // La fila del pixel central
        
        // Condicion para no considerar pixeles fuera de los limites de la imagen
        if ( (neighbour >= 0) && (neighbour < (N * N)) ) {
            center_neigh = global_id - (mid_row - neigh_row) * N;   // Se determina el indice del pixel 
                                                                    // central de cada fila de la imagen

            // Condicion para no considerar pixeles fuera de la vecindad
            if ( (neighbour >= (center_neigh - V)) && (neighbour <= (center_neigh + V)) ) {
                B[global_id] = B[global_id] + A[neighbour];         // Se suma el vecino directamente
                                                                    // al arreglo de memoria global
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__host__ int main(int argc, char** argv) {
    
    clock_t start_t, end_t; // Variables para catpurar el tiempo de ejecucion secuencial 
    float sum_gpu, sum_seq, gpu_time, cpu_time; // Variables para almacenar la suma y los tiempos de ejecucion finales
    char* nValue = (char*)malloc(sizeof(char));
    char* bValue = (char*)malloc(sizeof(char)); 
    char* vValue = (char*)malloc(sizeof(char));

    // Se capturan los parametros de entrada
    getParams (argc, argv, nValue, bValue, vValue);

    // Se transforman a numeros enteroros para su mejor manipulacion
    int N = atoi(nValue);   // Tamaño de la imagen
    int Bs = atoi(bValue);  // Tamaño de bloque
    int V = atoi(vValue);   // Radio de la vecindad

    dim3 gridSize = dim3(N / Bs, N / Bs);   // Se declara una grilla bidimensional dimension N/Bs x N/Bs
    dim3 blockSize = dim3(Bs, Bs);          // Se declaran bloques bidimensionales de Bs x Bs
    
    float* h_a = (float*)malloc( (N * N) * sizeof(float));  // Se aloja memoria para la imagen en host
    float* h_b = (float*)malloc( (N * N) * sizeof(float));  // Se aloja memoria para la imagen de salida en host
    float* seq_b = (float*)malloc( (N * N) * sizeof(float));// Se aloja memoria para la imagen de salida secuencial en host

    // Se declaran las variables para las variables de la imagen, y la imagen de salida en device
    float* d_a;
    float* d_b;

    // Se rellena la imagen con valores aleatorios entre 0 y 1
    for (int index = 0; index < (N * N); index++) {
        h_a[index] = (float) rand() / RAND_MAX; 
    }

    // Se crean y se inicializan los eventos para capturar el tiempo de ejecucion para 
    // todas las operaciones que se realizan utilizando la GPU.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Se aloja memoria para ls variables en device
    cudaMalloc((void**) &d_a, (N * N) * sizeof(float));
    cudaMalloc((void**) &d_b, (N * N) * sizeof(float));

    // Se traspasa el contenido desde host hacia device
    cudaMemcpy(d_a, h_a, (N * N) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, (N * N) * sizeof(float), cudaMemcpyHostToDevice);

    // Se ejecuta el kernel
    suma2D<<<gridSize, blockSize>>>(d_a, d_b, N, V);

    // Se traspasa el contenido desde device hacia host
    cudaMemcpy(h_b, d_b, (N * N) * sizeof(float), cudaMemcpyDeviceToHost);

    // Se detienen los eventos para obtener el tiempo de ejecucion final en GPU
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Se realiza la suma de los pixeles de la matriz resultante por el metodo paralelo
    sum_gpu = pixelSum (h_b, N);
    
    // Se muestran los tiempos de ejecucion y la suma final por consola
    printf("Tiempo GPU: %f (ms)\n", gpu_time);
    printf("Suma GPU: %f\n", sum_gpu);

    // Se inicia el reloj para obtener el tiempo de ejecuion de la solucion secuencial en CPU
    start_t = clock();

    // Se ejecuta la funcion secuencial
    suma2D_CPU (h_a, seq_b, N, V);

    // Se detiene el reloj y se obtiene el tiempo de ejecucion en milisegundos
    end_t = clock();
    cpu_time = (float)(end_t - start_t) / CLOCKS_PER_SEC;
    cpu_time *= 1000;

    // Se realiza la suma de los pixeles de la matriz resultante por el metodo secuencial
    sum_seq = pixelSum (seq_b, N);

    // Se muestran los tiempos de ejecucion y la suma final por consola
    printf("Tiempo CPU: %f (ms)\n", cpu_time);
    printf("Suma CPU: %f\n", sum_seq);

    // Destruccion de los eventos iniciados
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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

    // Se definen variables para realizar la suma de la vecindad de un determinado pixel de la imagen
    int index, offset, neighbour, mid_row, neigh_row, center_neigh;

    // Se recorre la imagen 
    for (index = 0; index < (N * N); index++){
        B[index] = 0.0; // Se inicializa en 0.0 cada elemento del arreglo de salida

        for (offset = -V * (1 + N); offset <= V * (1 + N); offset++) {
            neighbour = index + offset;
            neigh_row = neighbour / N;
            mid_row = index / N;
            
            // Condicion para no considerar vecinos fuera de los limites de la imagen
            if ( (neighbour >= 0) && (neighbour < (N * N)) ) {
                center_neigh = index - (mid_row - neigh_row) * N;
    
                // Condicion para no considerar vecinos fuera de la vecindad
                if ( (neighbour >= (center_neigh - V)) && (neighbour <= (center_neigh + V)) ) {
                    B[index] = B[index] + A[neighbour];
                }
            }
        } 
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// - INPUTS: - argc: Largo del arreglo de argumentos argv.
//           - argv: Arreglo con los argumentos de entrada incluyendo en nombre del archivo.
//           - nValue: Tamaño de la imagen de entrada
//           - bValue: Tamaño de bloque de entrada
//           - vValue: Radio de la vecindad
// - OUTPUTS: -

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
// - DESCRIPTION: Determina la suma de los elementos de una imagen, de largo y ancho N.

float pixelSum (float* image, int N) {

    int index;
    float sum = 0.0;

    for (index = 0; index < (N * N); index++) {
        sum += image[index];
    }

    return sum;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// - DESCRIPTION: Mostrar una matriz por consola

void printImage (float* image, int N) {

    for (int index = 0; index < (N * N); index++) {
        printf("%f ", image[index]);

        if ((index + 1) % N == 0)
            printf("\n");
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////// END ////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////