# include <getopt.h>
# include <ctype.h>
# include <stdlib.h>
# include <stdio.h>
# include <unistd.h>
# include <string.h>
# include <pmmintrin.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void suma2D_SHMEM (float* A, float* B, int N, int V) {
    
    int local_i = threadIdx.x;
    int local_j = threadIdx.y;

    int global_i = blockDim.x * blockIdx.x + local_i;
    int global_j = blockDim.y * blockIdx.y + local_j;

    int global_id = global_i + global_j * N;

    B[global_id] = 0.0;

    if (global_id < (N * N))
        B[global_id] = A[global_id] + 1;
    
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__host__ int main(int argc, char** argv) {
    
    int N = atoi(argv[1]);
    int Bs = atoi(argv[2]);
    int V = atoi(argv[3]);

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

    cudaFree(d_a);
    cudaFree(d_b);
 
    // Se libera la memoria del host
    free(h_a);
    free(h_b);

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




