#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define PI 3.14159265358979323846
#define MAX_RADII 50
#define Kn 9.9
#define THRESH 1000
#define angular_granularity 60
#define LOG false

#define FNAME "coins"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// __constant__ __device__ int img_params[2] = {1024, 768};
__constant__ __device__ int img_width = 1024;
__constant__ __device__  int img_height = 768;
__constant__ __device__ int d_a_vals[4] = {2, 4, 6, 8};
__constant__ __device__ int d_b_vals[4] = {2, 4, 6, 8};

const vector<int> a_vals = {2, 4, 6, 8};
const vector<int> b_vals = {2, 4, 6, 8};

int img_width_c, img_height_c;

typedef struct pii {
    int first;
    int second;
} pii;

typedef struct pdd {
    float first;
    float second;
} pdd;

typedef struct Pixel {
    uint8_t r,g,b;
} Pixel;

uint64_t micros()
{
    uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    return us; 
}

__device__ void matInv(float (&A)[2][2], float (&A_inv)[2][2]){
    float det = 1 / (A[0][0] * A[1][1] - A[0][1] * A[1][0]);
    A_inv[0][0] = A[1][1] * det;
    A_inv[0][1] = -A[0][1] * det;
    A_inv[1][0] = -A[1][0] * det;
    A_inv[1][1] = A[0][0] * det;
}

__device__ void M_mults(float (&M)[2][2], float (&res)[2][2]){
    float M_inv[2][2];
    matInv(M, M_inv);
    res[0][0] = M[0][0] * M_inv[1][1] - M[0][1] * M_inv[0][1];
    res[0][1] = M[0][1] * M_inv[0][0] - M[0][0] * M_inv[1][0] ;
    res[1][0] = M[1][0] * M_inv[1][1] - M[1][1] * M_inv[0][1];
    res[1][1] = M[1][1] * M_inv[0][0] - M[1][0] * M_inv[1][0];
}

typedef struct Image {
    int w, h;
    Pixel** img;
    Image(int width, int height) {
        w = width; 
        h = height;
        img = (Pixel**)calloc(h, sizeof(Pixel*) * h);
        for(int i = 0; i < h ;i++){
            img[i] = (Pixel*)calloc(w, sizeof(Pixel));
        }
    }
} Image;

__device__ int get_2d_idx(int i, int j){
    return (i * img_width) + j;
}

__global__ void ellipseResponseMapKernel(float *Mg, float *Og, float *gradX, float *gradY, float *gradients){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size_a = sizeof(d_a_vals) / sizeof(int);
    int size_b = sizeof(d_b_vals) / sizeof(int);
    int idx = y * img_width + x;
    if(y < img_height && y >= 0 && x < img_width && x >= 0){
        if(gradients[idx] != 0){

            float dy = gradY[idx] / gradients[idx];
            float dx = gradX[idx] / gradients[idx];
            for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++){
                int degr = degr_idx * angular_granularity;
                float theta = degr * PI / 180.0;
                for(int a_idx = 1; a_idx < size_a; a_idx++){
                    for(int b_idx = 1; b_idx < size_b; b_idx++){
                        // printf("(%d, %d) in (%d, %d, %d)\n", y, x, degr_idx, a_idx, b_idx);
                        int a = d_a_vals[a_idx];
                        int b = d_b_vals[b_idx];
                        float G[2][2] = {
                            {a * cos(theta), -b * sin(theta)},
                            {a * sin(theta),  b * cos(theta)},
                        };
                        float transform_matrix[2][2];
                        M_mults(G, transform_matrix);
                        float grad_t_y = dy * transform_matrix[0][0] + dx * transform_matrix[0][1];
                        float grad_t_x = dy * transform_matrix[1][0] + dx * transform_matrix[1][1];
                        // int idx5d = degr_idx * size_a * size_b * img_width * img_height +
                        //     a_idx * size_b * img_width * img_height +
                        //     b_idx * img_width * img_height +
                        //     y * img_width +
                        //     x;
                        // printf("(%d, %d), (%f, %f)\n", y, x, grad_t_y, grad_t_x);
                        for(int n = 1; n < MAX_RADII; n++){
                            int p_plus_y  = y + grad_t_y * n;
                            int p_plus_x  = x + grad_t_x * n;
                            // printf("(%d, %d)-(%d,%d) voted\n", y, x, p_plus_y, p_plus_x);
                            int p_minus_y = y - grad_t_y * n;
                            int p_minus_x = x - grad_t_x * n;
                            if(p_plus_y >= 0 && p_plus_y < img_height && p_plus_x >= 0 && p_plus_x < img_width){
                                int idx5d = degr_idx * size_a * size_b * img_width * img_height +
                                            a_idx * size_b * img_width * img_height +
                                            b_idx * img_width * img_height +
                                            p_plus_y * img_width +
                                            p_plus_x;
                                Mg[idx5d] += gradients[idx];
                            }
                            if(p_minus_y >= 0 && p_minus_y < img_height && p_minus_x >= 0 && p_minus_x < img_width){
                                int idx5d = degr_idx * size_a * size_b * img_width * img_height +
                                            a_idx * size_b * img_width * img_height +
                                            b_idx * img_width * img_height +
                                            p_minus_y * img_width +
                                            p_minus_x;
                                Mg[idx5d] -= gradients[idx];
                            }
                            // __syncwarp();
                        }
                    }
                }
            }
        }
            
    }
}

__global__ void checkMgExists(float *Mg){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size_a = sizeof(d_a_vals) / sizeof(int);
    int size_b = sizeof(d_b_vals) / sizeof(int);
    int idx = y * img_width + x;
    if(y < img_height && y >= 0 && x < img_width && x >= 0){
        float maxval = 0.0;
        for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++){
            for(int a_idx = 0; a_idx < size_a; a_idx++){
                for(int b_idx = 0; b_idx < size_b; b_idx++){
                    int idx5d = degr_idx * size_a * size_b * img_width * img_height +
                                        a_idx * size_b * img_width * img_height +
                                        b_idx * img_width * img_height +
                                        y * img_width +
                                        x;
                    if(Mg[idx5d] > maxval) maxval = Mg[idx5d];
                }
            }
        }
        if(maxval != 0.0)
            printf("(%d,%d) has response %f\n", y, x, maxval);
    }
}

__global__ void checkExists(float *S){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * img_width + x;
    if(y < img_height && y >= 0 && x < img_width && x >= 0){
        if(S[idx] != 0) 
            printf("(%d, %d) - %f\n", y, x, S[idx]);
    }
}

//gaussian blur the 5d feature maps
__global__ void generate_S1(float *M, float *O, float *S, float *S_nms,
                float *S_flat, float *S_flat_nms){
    float gauss[3][3] = {
        {1.0/16.0, 2.0/16.0, 1.0/16.0},
        {2.0/16.0, 4.0/16.0, 2.0/16.0},
        {1.0/16.0, 2.0/16.0, 1.0/16.0}
    };
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size_a = sizeof(d_a_vals) / sizeof(int);
    int size_b = sizeof(d_b_vals) / sizeof(int);
    for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++){
        for(int a_idx = 0; a_idx < size_a; a_idx++){
            for(int b_idx = 0; b_idx < size_b; b_idx++){
                if(y < (img_height - 1) && y > 0 && x < (img_width - 1) && x >= 1){
                    for(int dy = 0; dy < 3; dy++){
                        for(int dx = 0; dx < 3; dx++){
                            int idx =   degr_idx * size_a * size_b * img_width * img_height +
                                        a_idx * size_b * img_width * img_height +
                                        b_idx * img_width * img_height +
                                        y * img_width +
                                        x;
                            int idx2 =  degr_idx * size_a * size_b * img_width * img_height +
                                        a_idx * size_b * img_width * img_height +
                                        b_idx * img_width * img_height +
                                        (dy + y - 1) * img_width +
                                        (dx + x -1);
                            S[idx] += M[idx2] * gauss[dy][dx];
                        }
                    }
                }        
            }
        }
    }
}

//NMS the 5d feature maps 
__global__ void generate_S2(float *M, float *O, float *S, float *S_nms,
                float *S_flat, float *S_flat_nms){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size_a = sizeof(d_a_vals) / sizeof(int);
    int size_b = sizeof(d_b_vals) / sizeof(int);
    for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++){
        for(int a_idx = 0; a_idx < size_a; a_idx++){
            for(int b_idx = 0; b_idx < size_b; b_idx++){
                if(y < (img_height - 1) && y > 0 && x < (img_width - 1) && x >= 1){
                    float maxNeighbor = 0.0;
                    for(int dy = 0; dy < 3; dy++){
                        for(int dx = 0; dx < 3; dx++){        
                            int idx2 =  degr_idx * size_a * size_b * img_width * img_height +
                                        a_idx * size_b * img_width * img_height +
                                        b_idx * img_width * img_height +
                                        (dy + y - 1) * img_width +
                                        (dx + x -1);
                            if(dy != 1 || dx != 1) maxNeighbor = max(maxNeighbor, S[idx2]);
                        }
                    }
                    int idx = degr_idx * size_a * size_b * img_width * img_height +
                                a_idx * size_b * img_width * img_height +
                                b_idx * img_width * img_height +
                                y * img_width +
                                x;
                    if(S[idx] < maxNeighbor) S_nms[idx] = 0;
                    else S_nms[idx] = S[idx];
                }
                        
            }
        }
    }
    float maxresponse = 0.0;
    for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++){
        for(int a_idx = 0; a_idx < size_a; a_idx++){
            for(int b_idx = 0; b_idx < size_b; b_idx++){
                int idx = degr_idx * size_a * size_b * img_width * img_height +
                                a_idx * size_b * img_width * img_height +
                                b_idx * img_width * img_height +
                                y * img_width +
                                x;
                maxresponse = max(maxresponse, S[idx]);
            }
        }
    }
    S_flat_nms[y * img_width + x] = maxresponse;
}

__global__ void generate_S3(float *M, float *O, float *S, float *S_nms,
                float *S_flat, float *S_flat_nms){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size_a = sizeof(d_a_vals) / sizeof(int);
    int size_b = sizeof(d_b_vals) / sizeof(int);
    float gauss[3][3] = {
        {1.0/16.0, 2.0/16.0, 1.0/16.0},
        {2.0/16.0, 4.0/16.0, 2.0/16.0},
        {1.0/16.0, 2.0/16.0, 1.0/16.0}
    };
    if(y < (img_height - 1) && y > 0 && x < (img_width - 1) && x >= 1){
        S_flat[y * img_width + x] = 0.0;
        for(int dy = 0; dy < 3; dy++){
            for(int dx = 0; dx < 3; dx++){
                S_flat[y * img_width + x] += S_flat_nms[(dy+y-1)*img_width + dx+x-1] * gauss[dy][dx];
            }
        }
    }
    
}

__global__ void generate_S4(float *M, float *O, float *S, float *S_nms,
                float *S_flat, float *S_flat_nms){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size_a = sizeof(d_a_vals) / sizeof(int);
    int size_b = sizeof(d_b_vals) / sizeof(int);
    float maxNeighbor = 0.0;
    if(y < (img_height - 1) && y > 0 && x < (img_width - 1) && x >= 1){
        for(int dy = 0; dy < 3; dy++){
            for(int dx = 0; dx < 3; dx++){
                if(dy != 1 || dx != 1) maxNeighbor = max(maxNeighbor, S_flat[(dy+y-1)*img_width + dx+x-1]);
            }
    }
    }
    
    if(maxNeighbor > S_flat[y * img_width + x]) S_flat_nms[y * img_width + x] = 0;
    else S_flat_nms[y * img_width + x] = S_flat[y * img_width + x];
}

__global__ void grayscaleKernel(int *color_image_d, float *grayscaleImg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // if(!x && !y)
    
    if (x < img_width && y < img_height) {
        int idx = get_2d_idx(y,x);
        
        grayscaleImg[idx] = 0.2125 * color_image_d[3 * idx] 
                            + 0.7154 * color_image_d[3 * idx + 1] 
                            + 0.0721 * color_image_d[3 * idx + 2];
        if(grayscaleImg[idx] == 0){
            printf("(%d, %d) (%d,%d) wrote 0 to grayscale\n", y, x, img_height, img_width);
        }
        // printf("grayscale, %d, %d | %d, %d | %d ((%d,%d,%d)>%f)\n",
        //          y, x , img_height, img_width, idx, color_image_d[3 * idx],
        //          color_image_d[3 * idx + 1], color_image_d[3 * idx + 2],  grayscaleImg[idx]);
    }
    else {
        // printf("(%d, %d) (%d,%d) did not write to grayscale\n", y, x, img_height, img_width);
    }

    if(y == 0 && x == 3){
        int idx = get_2d_idx(y,x);
        printf("PRINT GRAY TEST, idx = %d, v = %f\n", idx , grayscaleImg[idx]);
    }
    // __syncthreads();
}

__global__ void gradientKernel(float *grayscaleImg, float *gradients, float *gradX, float *gradY, float *gradientDir) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int flat_idx = get_2d_idx(y, x);
    if (x > 0 && y > 0 && x < (img_width - 1) && y < (img_height - 1)) {
        float xSobel[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
        float ySobel[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
        float xg = 0, yg = 0;
        if(y == 504 && x == 590){
            printf("gray value is %f\n ", grayscaleImg[flat_idx]);
        }
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                float pixel = grayscaleImg[get_2d_idx(y + dy, x + dx)];
                xg += pixel * xSobel[dy + 1][dx + 1];
                yg += pixel * ySobel[dy + 1][dx + 1];
                if(y == 504 && x == 590){
                    printf("iterating through (%d, %d) gray =%f\n", y+dy, x+dx, pixel);
                }
                
            }
        }
        
        float mag = sqrt(xg * xg + yg * yg);
        gradients[flat_idx] = (mag > 200) ? mag : 0;
        if(y == 504 && x == 590){
            printf("(%d, %d) has grads, (%f, %f) = %f\n", y, x, xg, yg, mag);
        }
        gradX[flat_idx] = xg;
        gradY[flat_idx] = yg;
        float angle = atan(yg/xg);
        if(isnan(gradientDir[flat_idx])) gradientDir[flat_idx] = PI / 2 * ((yg > 0) - (yg < 0));
    }
    else {
        // printf("(%d, %d) (%d,%d) did not write to gradients\n", y, x, img_height, img_width);
    }
    // __syncthreads();
}

__global__ void NMSKernel(float *gradients, float *gradientDir, float *nms_gradients) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < img_width - 1 && y < img_height - 1) {
        int idx = y * img_width + x;
        float n1, n2;
        if (gradients[idx] != 0) {
            float dir = gradientDir[idx];
            if (dir > PI / 4 || dir < -PI / 4) {
                n1 = gradients[(y - 1) * img_width + x];
                n2 = gradients[(y + 1) * img_width + x];
            } else if (dir > 0) {
                n1 = gradients[(y - 1) * img_width + (x + 1)];
                n2 = gradients[(y + 1) * img_width + (x - 1)];
            } else if (dir > -PI / 4) {
                n1 = gradients[(y + 1) * img_width + (x + 1)];
                n2 = gradients[(y - 1) * img_width + (x - 1)];
            } else {
                n1 = 0;
                n2 = 0;
            }

            if (gradients[idx] >= n1 && gradients[idx] >= n2) {
                nms_gradients[idx] = gradients[idx];
            } else {
                nms_gradients[idx] = 0;
            }
        }
    }
    // __syncthreads();
}

__global__ void radialSymmetryKernel(float *gradX, float *gradY, float *gradients, float *O, float *M) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < img_width && y < img_height && gradients[y * img_width + x] != 0) {
        float dy = gradY[y * img_width + x] / gradients[y * img_width + x];
        float dx = gradX[y * img_width + x] / gradients[y * img_width + x];

        for (int r = 1; r < MAX_RADII; r++) {
            int p_plus_y = y + dy * r;
            int p_plus_x = x + dx * r;
            int p_minus_y = y - dy * r;
            int p_minus_x = x - dx * r;


            // jsut using atomic add but we can change it to the blocking method
            if (p_plus_y >= 0 && p_plus_y < img_height && p_plus_x >= 0 && p_plus_x < img_width) {
                atomicAdd(&O[r * img_height * img_width + p_plus_y * img_width + p_plus_x], 1);
                atomicAdd(&M[r * img_height * img_width + p_plus_y * img_width + p_plus_x], gradients[y * img_width + x]);
            }
            if (p_minus_y >= 0 && p_minus_y < img_height && p_minus_x >= 0 && p_minus_x < img_width) {
                atomicAdd(&O[r * img_height * img_width + p_minus_y * img_width + p_minus_x], -1);
                atomicAdd(&M[r * img_height * img_width + p_minus_y * img_width + p_minus_x], -gradients[y * img_width + x]);
            }
        }
    }
    // __syncthreads();
}

__global__ void gaussConvolve1(float *M, float *postGauss, int *radii){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float gauss[3][3] = {
        {1.0/16.0, 2.0/16.0, 1.0/16.0},
        {2.0/16.0, 4.0/16.0, 2.0/16.0},
        {1.0/16.0, 2.0/16.0, 1.0/16.0}
    };

    float t = 0;
    int best_r = 0;
    for(int r = 0; r < MAX_RADII; r++){
        if(M[r*img_width*img_height + y*img_width + x] > t){
            best_r = r;
            t = M[r*img_width*img_height + y*img_width + x];
        }
    }
    radii[y*img_width + x] = best_r;
    M[0*img_width*img_height + y*img_width + x] = t;
    // __syncthreads();
}


__global__ void gaussConvolve2(float *M, float *postGauss, int *radii){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float gauss[3][3] = {
        {1.0/16.0, 2.0/16.0, 1.0/16.0},
        {2.0/16.0, 4.0/16.0, 2.0/16.0},
        {1.0/16.0, 2.0/16.0, 1.0/16.0}
    };
    float g = 0;
    for(int dy = 0; dy < 3; dy++){
        for(int dx = 0; dx < 3; dx++){
            g += M[(y + dy) * img_width + (x + dx)] * gauss[dy][dx];
        }
    }
    postGauss[y * img_width + x] = (g > THRESH) ? g : 0;
    // __syncthreads();
}

__global__ void postGaussNMSKernel(float *postGauss, float *gaussNMS, int *radii) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < img_width - 1 && y < img_height - 1) {
        int idx = y *img_width + x;
        float maxNeighbor = 0.0;
        // fidn the max value in the block and keep track of it
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int neighbor_idx = (y + dy) * img_width + (x + dx);
                if (dy != 0 || dx != 0) {
                    if(maxNeighbor <= postGauss[neighbor_idx]){
                        maxNeighbor = postGauss[neighbor_idx];
                    }
                }
            }
        }
        // if this pixel is the max value, then keep it, otherwise delete
        if (postGauss[idx] >= maxNeighbor) {
            gaussNMS[idx] = postGauss[idx];
        } else {
            gaussNMS[idx] = 0;
            radii[idx] = 0;
        }
    }
    // __syncthreads();
}

//cpu
unsigned long
driverCuda(){
    string basefile = "coins";
    string filename = "images/" + basefile + ".txt";
    ifstream fin(filename);
    fin >> img_width_c >> img_height_c;
    printf("Base file %s\n", basefile.c_str());
    printf("Reading image path [%s] (%d x %d)\n", filename.c_str(), img_height_c, img_width_c);
    // cudaMemcpy(&img_height, &img_height_c, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(&img_width, &img_width_c, sizeof(int), cudaMemcpyHostToDevice);
    int* color_img_cpu = (int*)calloc(img_width_c * img_height_c * 3, sizeof(int));
    for(int i = 0; i < img_height_c; i++){
        for(int j = 0; j < img_width_c; j++){
            int a,b,c;
            fin >> a >> b >> c;
            int flat_idx = (i * img_width_c + j) * 3;
            color_img_cpu[flat_idx] = a;
            color_img_cpu[flat_idx + 1] = b;
            color_img_cpu[flat_idx + 2] = c; 
        }
    }

    // unsigned long start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    //determine kernel call params
    dim3 blockDim(32,32, 1);
    dim3 gridDim((img_width_c + blockDim.x - 1) / blockDim.x,
                 (img_height_c + blockDim.y - 1) / blockDim.y,
                 1);
    // dim3 gridDim(1, 1, 1);


    //initialzie data structures
    float *grayscaleImage_d; int *color_img_d;
    int nBytesToCopy = img_width_c*img_height_c*3*sizeof(int);
    printf("transferring %d bytes\n", nBytesToCopy);
    unsigned long start = micros();
    cudaMalloc(&color_img_d, img_width_c*img_height_c*3*sizeof(int));
    unsigned long device_transfer_start = micros();
    cudaMemcpy(color_img_d, color_img_cpu, img_width_c*img_height_c*3*sizeof(int),  cudaMemcpyHostToDevice);
    unsigned long device_transfer_end = micros();

    unsigned long device_malloc_start = micros();
    cudaMalloc(&grayscaleImage_d, img_width_c * img_height_c * sizeof(float));

    float *gradients_d, *gradientDir_d, *gradX_d, *gradY_d, *nms_gradients_d;
    cudaMalloc(&gradients_d, img_width_c * img_height_c * sizeof(float));
    cudaMalloc(&gradientDir_d, img_width_c * img_height_c * sizeof(float));
    cudaMalloc(&gradX_d, img_width_c * img_height_c * sizeof(float));
    cudaMalloc(&gradY_d, img_width_c * img_height_c * sizeof(float));
    cudaMalloc(&nms_gradients_d, img_width_c * img_height_c * sizeof(float));

    float *O_d, *M_d;
    cudaMalloc(&O_d, MAX_RADII * img_width_c * img_height_c * sizeof(float));
    cudaMalloc(&M_d, MAX_RADII * img_width_c * img_height_c * sizeof(float));

    float *postGauss_d, *gaussNMS_d; int *radii_d;
    cudaMalloc(&postGauss_d, img_width_c * img_height_c * sizeof(float));
    cudaMalloc(&gaussNMS_d, img_width_c * img_height_c * sizeof(float));
    cudaMalloc(&radii_d,     img_width_c * img_height_c * sizeof(int));

    float *Mg_d, *Og_d, *S_d, *S_nms_d, *S_flat_d, *S_nms_flat_d;
    cudaMalloc(&Mg_d, img_width_c * img_height_c * (360/angular_granularity) * a_vals.size() * b_vals.size() * sizeof(float));
    cudaMalloc(&Og_d, img_width_c * img_height_c * (360/angular_granularity) * a_vals.size() * b_vals.size() * sizeof(float));
    cudaMalloc(&S_d, img_width_c * img_height_c * (360/angular_granularity) * a_vals.size() * b_vals.size() * sizeof(float));
    cudaMalloc(&S_nms_d, img_width_c * img_height_c * (360/angular_granularity) * a_vals.size() * b_vals.size() * sizeof(float));
    cudaMalloc(&S_flat_d, img_width_c * img_height_c * sizeof(float));
    cudaMalloc(&S_nms_flat_d, img_width_c * img_height_c * sizeof(float));
    unsigned long device_malloc_end = micros();

    //cpu mallocs
    float *postGauss_c, *gaussNMS_c, *S_flat_c, *S_flat_nms_c;
    int *radii_c;
    postGauss_c = (float*) malloc(img_width_c * img_height_c * sizeof(float));
    gaussNMS_c  = (float*) malloc(img_width_c * img_height_c * sizeof(float));
    radii_c     = (int*) malloc(img_width_c * img_height_c * sizeof(int));
    S_flat_c = (float*) malloc(img_width_c * img_height_c * sizeof(float));
    S_flat_nms_c = (float*) malloc(img_width_c * img_height_c * sizeof(float));


    //grayscale the image
    printf("grayscale start\n");
    unsigned long gray_start = micros();
    grayscaleKernel<<<gridDim, blockDim>>>(color_img_d, grayscaleImage_d);
    cudaThreadSynchronize();
    unsigned long gray_end = micros();
    printf("grayscale end\n");

    if(LOG){
        float *grayscale_c = (float*) malloc(img_height_c * img_width_c * sizeof(float));
        cudaMemcpy(grayscale_c, grayscaleImage_d, img_width_c * img_height_c * sizeof(float), cudaMemcpyDeviceToHost);
        ofstream gray_file("images_cuda/" + basefile + "_gray.txt");
        for(int i = 0; i < img_height_c; i++){
            for(int j = 0; j < img_width_c; j++){
                gray_file << grayscale_c[i * img_width_c + j] << endl;
            }
        }
        gray_file.close();
    }

    printf("gradient calculation start\n");
    unsigned long gradient_start = micros();
    gradientKernel<<<gridDim, blockDim>>>(grayscaleImage_d, gradients_d, gradX_d, gradY_d, gradientDir_d);
    cudaThreadSynchronize();
    unsigned long gradient_end = micros();
    printf("gradient calculation end\n");

    if(LOG){
        float *gradients_c = (float*) malloc(img_height_c * img_width_c * sizeof(float));
        cudaMemcpy(gradients_c, gradients_d, img_width_c * img_height_c * sizeof(float), cudaMemcpyDeviceToHost);
        ofstream grad_file("images_cuda/" + basefile + "_gradients.txt");
        for(int i = 0; i < img_height_c; i++){
            for(int j = 0; j < img_width_c; j++){
                grad_file << gradients_c[i * img_width_c + j] << endl;
            }
        }
        grad_file.close();
    }

    printf("NMS start\n");
    unsigned long nms_start = micros();
    NMSKernel<<<gridDim,blockDim>>>(gradients_d, gradientDir_d, nms_gradients_d);
    cudaThreadSynchronize();
    unsigned long nms_end = micros();
    printf("NMS end\n");

    printf("radial symmetry calculation start\n");
    unsigned long radial_sym_start = micros();
    radialSymmetryKernel<<<gridDim,blockDim>>>(gradX_d, gradY_d, gradients_d, O_d, M_d);
    cudaThreadSynchronize();
    unsigned long radial_sym_end = micros();
    printf("radial symmetry calculation end\n");

    printf("best response calculation start\n");
    unsigned long blur_start = micros();
    gaussConvolve1<<<gridDim, blockDim>>>(M_d, postGauss_d, radii_d);
    cudaThreadSynchronize();
    printf("best response calculation end\n");

    printf("blur start\n");
    gaussConvolve2<<<gridDim, blockDim>>>(M_d, postGauss_d, radii_d);
    cudaThreadSynchronize();
    unsigned long blur_end = micros();
    printf("blur end\n");

    unsigned long transfer1start = micros();
    cudaMemcpy(postGauss_c, postGauss_d, img_width_c * img_height_c * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(radii_c, radii_d, img_width_c * img_height_c * sizeof(int), cudaMemcpyDeviceToHost);
    unsigned long transfer1time = micros() - transfer1start;

    printf("NMS on best-radii start\n");
    unsigned long circle_nms_start = micros();
    postGaussNMSKernel<<<gridDim, blockDim>>>(postGauss_d, gaussNMS_d, radii_d);
    cudaThreadSynchronize();
    unsigned long circle_nms_end = micros();
    printf("NMS on best-radii end\n");

    cudaMemcpy(gaussNMS_c, gaussNMS_d, img_width_c * img_height_c * sizeof(float), cudaMemcpyDeviceToHost);

    /*
    
        ELLIPSES FROM HERE, CIRCLES BEFORE

    */

    printf("ellipse response map start\n");
    unsigned long ellipse_start = micros();
    // ellipseResponseMapKernel<<<gridDim,blockDim>>>(Mg_d, Og_d, gradX_d, gradY_d, nms_gradients_d, gradientDir_d);
    ellipseResponseMapKernel<<<gridDim, blockDim>>>(Mg_d, Og_d, gradX_d, gradY_d, nms_gradients_d);
    cudaThreadSynchronize();
    unsigned long ellipse_end = micros();
    printf("ellipse resposne map end\n");

    // printf("checking maxes start\n");
    // checkMgExists<<<gridDim, blockDim>>>(Mg_d);
    // cudaThreadSynchronize();
    // printf("checking maxes end\n");




    printf("S1 start\n");
    unsigned long ellipse_proc_start = micros();
    generate_S1<<<gridDim, blockDim>>>(Mg_d, Og_d, S_d, S_nms_d, S_flat_d, S_nms_flat_d);
    cudaThreadSynchronize();
    printf("S1 end\n");

    // printf("checking maxes start\n");
    // checkMgExists<<<gridDim, blockDim>>>(S_d);
    // cudaThreadSynchronize();
    // printf("checking maxes end\n");

    printf("S2 start\n");
    generate_S2<<<gridDim, blockDim>>>(Mg_d, Og_d, S_d, S_nms_d, S_flat_d, S_nms_flat_d);
    cudaThreadSynchronize();
    printf("S2 end\n");

    printf("S3 start\n");
    generate_S3<<<gridDim, blockDim>>>(Mg_d, Og_d, S_d, S_nms_d, S_flat_d, S_nms_flat_d);
    cudaThreadSynchronize();
    printf("S3 end\n");

    
    // printf("checking maxes start\n");
    // checkExists<<<gridDim, blockDim>>>(S_flat_d);
    // cudaThreadSynchronize();
    // printf("checking maxes end\n");

    printf("S4 start\n");
    generate_S4<<<gridDim, blockDim>>>(Mg_d, Og_d, S_d, S_nms_d, S_flat_d, S_nms_flat_d);
    cudaThreadSynchronize();
    unsigned long ellipse_proc_end = micros();
    printf("S4 end\n");



    


    // cudaMemcpy(postGauss_c, postGauss_d, img_width_c * img_height_c * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(gaussNMS_c, gaussNMS_d, img_width_c * img_height_c * sizeof(float), cudaMemcpyDeviceToHost);
    unsigned long host_transfer_start = micros();
    cudaMemcpy(radii_c, radii_d, img_width_c * img_height_c * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(S_flat_c, S_flat_d, img_width_c * img_height_c * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(S_flat_nms_c, S_nms_flat_d, img_width_c * img_height_c * sizeof(float), cudaMemcpyDeviceToHost);
    unsigned long host_transfer_end = micros();
    unsigned long end = micros();

    if(LOG){
        ofstream post_gauss_nms("images_cuda/" + basefile + "_gaussnms.txt");
        for(int i = 0; i < img_height_c; i++){
            for(int j = 0; j < img_width_c; j++){
                post_gauss_nms << gaussNMS_c[i * img_width_c + j] << endl;
            }
        }
        post_gauss_nms.close();

        ofstream radii_file("images_cuda/" + basefile + "_radii.txt");
        for(int i = 0; i < img_height_c; i++){
            for(int j = 0; j < img_width_c; j++){
                radii_file << radii_c[i * img_width_c + j] << endl;
            }
        }
        radii_file.close();

        ofstream blurred_file("images_cuda/" + basefile + "_genblurred.txt");
        for(int i = 0; i < img_height_c; i++){
            for(int j = 0; j < img_width_c; j++){
                // if(S_flat_c[i * img_width_c + j] != 0) printf("(%d,%d)\n", i, j);
                blurred_file << S_flat_c[i * img_width_c + j] << endl;
            }
        }
        blurred_file.close();

        ofstream gen_nms_file("images_cuda/" + basefile + "_gennms.txt");
        for(int i = 0; i < img_height_c; i++){
            for(int j = 0; j < img_width_c; j++){
                if(S_flat_nms_c[i * img_width_c + j] != 0) printf("(%d,%d)\n", i, j);
                gen_nms_file << S_flat_nms_c[i * img_width_c + j] << endl;
            }
        }
        gen_nms_file.close();
    }
    printf("to device transfer %ld\n", device_transfer_end-device_transfer_start);
    printf("device alloc %ld\n", device_malloc_end- device_malloc_start);
    printf("grayscale %ld\n", gray_end - gray_start);
    printf("gradient calc %ld\n", gradient_end - gradient_start);
    printf("gradient nms %ld\n", nms_end - nms_start);
    printf("radial symm %ld\n", radial_sym_end - radial_sym_start);
    printf("circle blur %ld\n", blur_end - blur_start);
    printf("circle nms %ld\n", circle_nms_end - circle_nms_start);
    printf("ellipse calc %ld\n", ellipse_end - ellipse_start);
    printf("ellipse process %ld\n", ellipse_proc_end - ellipse_proc_start);
    printf("to host transfer %ld\n", host_transfer_end - host_transfer_start + transfer1time);
    return end - start;
}


int main(){
    unsigned long runtime = driverCuda();
    printf("Runtime: %ld\n", runtime);
}