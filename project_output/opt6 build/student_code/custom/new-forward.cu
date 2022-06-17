#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define TILE_WIDTH1 24
#define TILE_WIDTH2 16

#define BLOCK_SIZE 1024


// Optimization 1 - Tiled Shared Memory convolution
__global__ void conv_forward_kernel_shared_mem(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    const int H_grid = ceil(1.0 * H_out / TILE_WIDTH);

    extern __shared__ float sArray[];
    float * sharedX = &sArray[0];
    int twk = TILE_WIDTH + K - 1;
    float * sharedKernel = &sArray[twk * twk];

    int n = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;

    int base_h = (blockIdx.z / W_grid) * TILE_WIDTH;
    int base_w = (blockIdx.z % W_grid) * TILE_WIDTH;
    
    float acc = 0;

    for (int c = 0; c < C; c++) {

        if (threadIdx.x < K && threadIdx.y < K ) {
            sharedKernel[threadIdx.y*K + threadIdx.x] = k4d(m,c,threadIdx.y,threadIdx.x);
        }

        __syncthreads();

        for (int i = h; i < base_h + twk; i+=TILE_WIDTH) {
            for (int j = w; j < base_w + twk; j+=TILE_WIDTH) {
                if (i < H && j < W) {
                    sharedX[(i - base_h) * twk + (j - base_w)] = x4d(n, c, i,j);
                } else {
                    sharedX[(i - base_h) * twk + (j - base_w)] = 0;
                }

            }

        }

        __syncthreads();

        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                if (threadIdx.y + p < twk && threadIdx.x + q < twk) {
                    acc += sharedX[(threadIdx.y + p) * twk + (threadIdx.x + q)] * sharedKernel[p * K + q]; 
                }    
            }
        }

        __syncthreads();

    }
    
    if (n < B && m < M && h < H_out && w < W_out) {

        y4d(n,m,h,w) = acc;
    }

    #undef y4d
    #undef x4d
    #undef k4d
    
}

// Optimization 2 - Shared memory matrix multiplication and input matrix unrolling 

__global__ void unroll_kernel(const float *x, float *unroll_x, int b, const int B, const int C, const int H, const int W, const int K, int H_out, int W_out, int unroll_H, int unroll_W) {
    
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
      
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx1 < C * unroll_W && idx2 < (B - b)) {

        int c = idx1 / unroll_W;
        int h_out = (idx1 % unroll_W) / W_out;
        int w_out = (idx1 % unroll_W) % W_out;
        
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int h_un = c * K * K + p * K + q;
                int w_un = h_out * W_out + w_out;
                // X_unroll(idx2, h_un, w_un) = x4d(idx2 + b, c, h_out + p, w_out + q);
                unroll_x[(idx2) * (unroll_H * unroll_W) + (h_un) * (H_out * W_out) + w_un] = x4d(idx2 + b, c, h_out + p, w_out + q);
            }
        }
    }
    #undef x4d
}

__global__ void matrixMultiplyShared(float *unrolled_x, float *y, const float *k, int b, const int B, const int M, const int C, const int H, const int W, const int K, int H_out, int W_out, int unroll_H, int unroll_W) 
{

__shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
__shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

int bx = blockIdx.x;
int by = blockIdx.y;
int bz = blockIdx.z;
int tx = threadIdx.x;
int ty = threadIdx.y;
int tz = threadIdx.z;

int row = by * TILE_WIDTH + ty;
int col = bx * TILE_WIDTH + tx;
int depth = bz * blockDim.z + tz;

int h_bound = unroll_H;

float Cvalue = 0;

if (depth < B) {

    for (int q = 0; q < ceil(1.0 * (h_bound) /  TILE_WIDTH); q++) {

        if (q * TILE_WIDTH + tx < unroll_H && row < M ) {
            subTileA[ty][tx] = k[row * unroll_H + q * TILE_WIDTH + tx];
        } else {
            subTileA[ty][tx] = 0.0;
        }

        if ((q * TILE_WIDTH + ty) < unroll_H && col < unroll_W ) {
            subTileB[ty][tx] = unrolled_x[(depth * unroll_W * C * K * K) + (q * TILE_WIDTH + ty) * unroll_W + col];
        } else {
            subTileB[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; j++) {
            Cvalue += subTileA[ty][j] * subTileB[j][tx];
        }
        
        __syncthreads();

        if (row < M && col < unroll_W && depth < B - b) {
            y[ (depth + b) * M * unroll_W + row * unroll_W + col] = Cvalue;
        }
    }
}
}

__global__ void matrixMultiplyShared2(float *A, float *B, float *C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns) {

    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
int bx = blockIdx.x;
int by = blockIdx.y;
int tx = threadIdx.x;
int ty = threadIdx.y;
int idx1 = by * blockDim.y + ty;
int idx2 = bx * blockDim.x + tx;

float Cvalue = 0;

for (int q = 0; q < numAColumns/TILE_WIDTH; q++) {

    int col = q * TILE_WIDTH + tx;
    int row = q * TILE_WIDTH + ty;

    if (col < numAColumns) {
        subTileA[ty][tx] = A[idx1 * numAColumns + col];
    } else {
        subTileA[ty][tx] = 0;
    }

    if (row < numAColumns) {
        subTileB[ty][tx] = B[(row) * numCColumns + idx2];
    } else {
        subTileB[ty][tx] = 0;
    }
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++) {
        Cvalue += subTileA[ty][k] * subTileB[k][tx];
    }
    __syncthreads();
}
if (idx1 < numCRows && idx2 < numCColumns) {
    C[ idx1 * numCColumns + idx2] = Cvalue;
}
}

// OPTIMIZATION 3 - tuning with restrict and loop unrolling
__global__ void fusedKernel(int M, int C, int H, int W, int K, int H_unroll, int W_unroll, const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ k) {
   
    int b = blockIdx.z * C * H * W;
    int area = H * W;

    #define x4d(i3, i2, i1, i0) x[b + (i2) * (area) + (i1) * (W) + i0]

    __shared__ float tileK[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileX[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int ksize = pow(K,2);
    int row_dist = col / W_out;
	int col_dist = col % W_out;

    float Cvalue = 0;

	int num_tiles = ceil(1.0 * H_unroll / TILE_WIDTH) * TILE_WIDTH;

    for(int i = 0; i < num_tiles; i+=TILE_WIDTH) {
		int col_index = (i + tx);
        if(row < M && col_index < H_unroll) {
            tileK[ty][tx] = k[row * H_unroll + col_index];
        } else {
            tileK[ty][tx] = 0;
        }

		int row_index = (i + ty);
        
        if(col < W_unroll && row_index < H_unroll) {
			int xc = row_index / ksize;
			int xs = row_index % ksize;

			int xrow = (xs / K) + row_dist;
			int xcol = (xs % K) + col_dist;

            tileX[ty][tx] = x4d(blockIdx.z, xc, xrow, xcol);
        } else {
            tileX[ty][tx] = 0;
        }
    __syncthreads();
    #pragma unroll
    for(int j = 0; j < TILE_WIDTH; j++) {
        Cvalue += tileK[ty][j] * tileX[j][tx];
    }
    __syncthreads();
  }
  if(row < M && col < W_unroll) {
    y[blockIdx.z * (M * H_out * W_out) + row * W_unroll + col] = Cvalue;
  }
  #undef x4d
}

// OPTIMIZATION 3 and 5 - Kernels for unrolling + restrict + different layer sizes 

__global__ void fusedKernel1(int M, int C, int H, int W, int K, int H_unroll, int W_unroll, const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ k) {
   
    int b = blockIdx.z * C * H * W;
    int area = H * W;

    #define x4d(i3, i2, i1, i0) x[b + (i2) * (area) + (i1) * (W) + i0]

    __shared__ float tileK[TILE_WIDTH1][TILE_WIDTH1];
    __shared__ float tileX[TILE_WIDTH1][TILE_WIDTH1];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int ksize = pow(K,2);
    int row_dist = col / W_out;
	int col_dist = col % W_out;

    float Cvalue = 0;

	int num_tiles = ceil(1.0 * H_unroll / TILE_WIDTH1) * TILE_WIDTH1;

    for(int i = 0; i < num_tiles; i+=TILE_WIDTH1) {
		int col_index = (i + tx);
        if(row < M && col_index < H_unroll) {
            tileK[ty][tx] = k[row * H_unroll + col_index];
        } else {
            tileK[ty][tx] = 0;
        }

		int row_index = (i + ty);
        
        if(col < W_unroll && row_index < H_unroll) {
			int xc = row_index / ksize;
			int xs = row_index % ksize;

			int xrow = (xs / K) + row_dist;
			int xcol = (xs % K) + col_dist;

            tileX[ty][tx] = x4d(blockIdx.z, xc, xrow, xcol);
        } else {
            tileX[ty][tx] = 0;
        }
    __syncthreads();
    #pragma unroll
    for(int j = 0; j < TILE_WIDTH1; j++) {
        Cvalue += tileK[ty][j] * tileX[j][tx];
    }
    __syncthreads();
  }
  if(row < M && col < W_unroll) {
    y[blockIdx.z * (M * H_out * W_out) + row * W_unroll + col] = Cvalue;
  }
  #undef x4d
}

__global__ void fusedKernel2(int M, int C, int H, int W, int K, int H_unroll, int W_unroll, const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ k) {
   
    int b = blockIdx.z * C * H * W;
    int area = H * W;

    #define x4d(i3, i2, i1, i0) x[b + (i2) * (area) + (i1) * (W) + i0]

    __shared__ float tileK[TILE_WIDTH2][TILE_WIDTH2];
    __shared__ float tileX[TILE_WIDTH2][TILE_WIDTH2];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int ksize = pow(K,2);
    int row_dist = col / W_out;
	int col_dist = col % W_out;

    float Cvalue = 0;

	int num_tiles = ceil(1.0 * H_unroll / TILE_WIDTH2) * TILE_WIDTH2;

    for(int i = 0; i < num_tiles; i+=TILE_WIDTH2) {
		int col_index = (i + tx);
        if(row < M && col_index < H_unroll) {
            tileK[ty][tx] = k[row * H_unroll + col_index];
        } else {
            tileK[ty][tx] = 0;
        }

		int row_index = (i + ty);
        
        if(col < W_unroll && row_index < H_unroll) {
			int xc = row_index / ksize;
			int xs = row_index % ksize;

			int xrow = (xs / K) + row_dist;
			int xcol = (xs % K) + col_dist;

            tileX[ty][tx] = x4d(blockIdx.z, xc, xrow, xcol);
        } else {
            tileX[ty][tx] = 0;
        }
    __syncthreads();
    #pragma unroll
    for(int j = 0; j < TILE_WIDTH2; j++) {
        Cvalue += tileK[ty][j] * tileX[j][tx];
    }
    __syncthreads();
  }
  if(row < M && col < W_unroll) {
    y[blockIdx.z * (M * H_out * W_out) + row * W_unroll + col] = Cvalue;
  }
  #undef x4d
}



// OPTIMIZATION 4 - Sweeping parameters
__global__ void sweep_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    const int H_grid = ceil(1.0 * H_out / TILE_WIDTH);

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


    int n = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    
    float acc = 0;

    if (h < H_out && w < W_out) {
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                        acc += x4d(n, c, h+p, w+q) * k4d(m,c,p,q);
                }
            }
        }
        y4d(n,m,h,w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}

__global__ void sweep_kernel1(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int W_grid = ceil(1.0 * W_out / TILE_WIDTH1);
    const int H_grid = ceil(1.0 * H_out / TILE_WIDTH1);

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


    int n = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH1 + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH1 + threadIdx.x;
    
    float acc = 0;

    if (h < H_out && w < W_out) {
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                        acc += x4d(n, c, h+p, w+q) * k4d(m,c,p,q);
                }
            }
        }
        y4d(n,m,h,w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}

__global__ void sweep_kernel2(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int W_grid = ceil(1.0 * W_out / TILE_WIDTH2);
    const int H_grid = ceil(1.0 * H_out / TILE_WIDTH2);

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


    int n = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH2 + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH2 + threadIdx.x;
    
    float acc = 0;

    if (h < H_out && w < W_out) {
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                        acc += x4d(n, c, h+p, w+q) * k4d(m,c,p,q);
                }
            }
        }
        y4d(n,m,h,w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}


// BASELINE
__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    const int H_grid = ceil(1.0 * H_out / TILE_WIDTH);

    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    // int n,m,h,w,c,p,q;

    int n = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.x;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.y;
    
    float acc = 0;

    if (h < H_out && w < W_out) {
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    // if (h + p >= 0 && h + p < H && w + q >= 0 && w + q < W) {
                        acc += x4d(n, c, h+p, w+q) * k4d(m,c,p,q);
                    // }
                }
            }
        }
        y4d(n,m,h,w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    // int Z = H_out * W_out;
    
    int sizeX = B * C * H * W * sizeof(float);
    int sizeY = B * M * H_out * W_out * sizeof(float);
    int sizeK = M * C * K * K * sizeof(float);

    cudaMalloc((void **) device_y_ptr, sizeY);
    cudaMalloc((void **) device_x_ptr, sizeX);
    cudaMalloc((void **) device_k_ptr, sizeK); 

    // int unroll_H = C * K * K;
    // int unroll_W = H_out * W_out;

    // cudaMalloc((void**) &unrolled_x, unroll_H * unroll_W * sizeof(float));

    cudaMemcpy(*device_y_ptr, host_y, sizeY, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_x_ptr, host_x, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr, host_k, sizeK, cudaMemcpyHostToDevice);

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    const int H_grid = ceil(1.0 * H_out / TILE_WIDTH);

    int Z = H_grid * W_grid;

    // dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    int X = M * W_out;
    int Y = B * H_out;
    // dim3 blockDim(X,Y,1);
    // dim3 gridDim(B,M,Z);

    // BASELINE
    // conv_forward_kernel<<<gridDim, blockDim>>>(device_y,device_x,device_k,B,M,C,H,W,K);

    // OPTIMIZATION 1
    // size_t shared_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);
    // conv_forward_kernel_shared_mem<<<gridDim, blockDim, shared_size>>>(device_y,device_x,device_k,B,M,C,H,W,K);

    // OPTIMIZATION 2 
    // float *unrolled_x;
    int unroll_H = C * K * K;
    int unroll_W = H_out * W_out;

    // int b_12 = B / 2;

    // cudaMalloc((void**) &unrolled_x, b_12 * unroll_H * unroll_W * sizeof(float));

    // dim3 blockDim1(TILE_WIDTH,TILE_WIDTH,1);
    // dim3 gridDim1(ceil((float) C * H_out * W_out / TILE_WIDTH), ceil((float) b_12 / TILE_WIDTH), 1);

    // dim3 blockDim2(TILE_WIDTH,TILE_WIDTH,1);
    // dim3 gridDim2(ceil((float) unroll_W / TILE_WIDTH), ceil((float) M / TILE_WIDTH), b_12);

    // for (int b = 0; b < B; b++) {
    //     unroll_kernel<<<gridDim1, blockDim1>>>(device_x, unrolled_x,b,C,H,W,K,H_out, W_out,unroll_H, unroll_W );
    //     matrixMultiplyShared<<<gridDim2, blockDim2>>>(unrolled_x, device_y, device_k, b, M, C, H, W, K, H_out, W_out,unroll_H, unroll_W );
    // }

    // unroll_kernel<<<gridDim1, blockDim1>>>(device_x, unrolled_x,  0, b_12, C, H,  W, K,  H_out,  W_out,  unroll_H,  unroll_W);
    // matrixMultiplyShared<<<gridDim2, blockDim2>>>(unrolled_x, device_y, device_k, 0,B, M, C, H, W, K, H_out, W_out,unroll_H, unroll_W );

    // cudaFree(unrolled_x);

    // cudaMalloc((void**) &unrolled_x, b_12 * unroll_H * unroll_W * sizeof(float));

    // unroll_kernel<<<gridDim1, blockDim1>>>(device_x, unrolled_x,b_12,B,C,H,W,K,H_out, W_out,unroll_H, unroll_W );
    // matrixMultiplyShared<<<gridDim2, blockDim2>>>(unrolled_x, device_y, device_k, b_12,B, M, C, H, W, K, H_out, W_out,unroll_H, unroll_W );

    // cudaFree(unrolled_x);

    // OPTIMIZATION 3

    // dim3 blockDim3(TILE_WIDTH,TILE_WIDTH,1);
    // dim3 gridDim3(ceil((float) unroll_W / TILE_WIDTH), ceil((float) M / TILE_WIDTH), B);

    // fusedKernel<<<gridDim3,blockDim3>>>( M, C, H, W, K, unroll_H, unroll_W, device_x, device_y, device_k);

    
    // OPTIMIZATION 5   

    // if (unroll_W < 1000) {
    //     dim3 blockDim5(TILE_WIDTH1,TILE_WIDTH1,1);
    //     dim3 gridDim5(ceil((float) unroll_W / TILE_WIDTH1), ceil((float) M / TILE_WIDTH1), B);

    //     fusedKernel1<<<gridDim5,blockDim5>>>( M, C, H, W, K, unroll_H, unroll_W, device_x, device_y, device_k);
    // } else {
    //     dim3 blockDim5(TILE_WIDTH2,TILE_WIDTH2,1);
    //     dim3 gridDim5(ceil((float) unroll_W / TILE_WIDTH2), ceil((float) M / TILE_WIDTH2), B);

    //     fusedKernel2<<<gridDim5,blockDim5>>>( M, C, H, W, K, unroll_H, unroll_W, device_x, device_y, device_k);
    // }

    // OPTIMIZATION 6  

    if (unroll_W < 1000) {
        dim3 blockDim5(TILE_WIDTH1,TILE_WIDTH1,1);
        dim3 gridDim5(B,M,Z);
    
        sweep_kernel1<<<gridDim5, blockDim5>>>(device_y,device_x,device_k,B,M,C,H,W,K);
    } else {
        dim3 blockDim5(TILE_WIDTH2,TILE_WIDTH2,1);
        dim3 gridDim5(B,M,Z);
    
        sweep_kernel2<<<gridDim5, blockDim5>>>(device_y,device_x,device_k,B,M,C,H,W,K);
    }

    
    // OPTIMIZATION 4
    // sweep_kernel<<<gridDim, blockDim>>>(device_y,device_x,device_k,B,M,C,H,W,K);
    

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int sizeY = B * M * H_out * W_out * sizeof(float);

    cudaMemcpy(host_y, device_y, sizeY, cudaMemcpyDeviceToHost);

    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);

    // Free device memory

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
