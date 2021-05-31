//nvcc cuda+mpi.cu -lmpi
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

using namespace std;

__global__ void matmul(float *A, float *B, float *C, int N){
    int i = blockIdx.y;
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    float sum = 0;
    for (int k=0; k<N; k++){
        sum += A[N*i+k] * B[N*k+j];
    }
    C[N*i+j] = sum;
}

int main(int argc, char** argv) {

  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int N = 2048;
  const int M = 1024;
  int gpusize = N * N * sizeof(float);

  float *A, *B, *C;
  cudaMallocManaged(&A, gpusize);
  cudaMallocManaged(&B, gpusize);
  cudaMallocManaged(&C, gpusize);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
      C[N*i+j] = 0;
    }
  }
  dim3 grid(N/M, N);

  double comp_time = 0, comm_time = 0;

  for(int irank=0; irank<size; irank++) {
    auto tic = chrono::steady_clock::now();
    matmul<<<grid, M>>>(A,B,C,N);
    cudaDeviceSynchronize();
    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();
    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();
  }
  

  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];

  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);
  if(rank==0) {
    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
  }

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  MPI_Finalize();
}
