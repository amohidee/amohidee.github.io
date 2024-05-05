#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define PI 3.14159265358979323846
#define MAX_RADII 50
#define Kn 9.9
#define THRESH 1000
#define angular_granularity 360
#define LOG true

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

__device__ void matInv(double (&A)[2][2], double (&A_inv)[2][2]){
    float det = 1 / (A[0][0] * A[1][1] - A[0][1] * A[1][0]);
    A_inv[0][0] = A[1][1] * det;
    A_inv[0][1] = -A[0][1] * det;
    A_inv[1][0] = -A[1][0] * det;
    A_inv[1][1] = A[0][0] * det;
}

__device__ void M_mults(double (&M)[2][2], double (&res)[2][2]){
    double M_inv[2][2];
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

// __global__ void aellipseResponseMapKernel(double *Mg, double *Og, double *gradX, double *gradY, double *gradients, double *gradientDir){
//     printf("x\n");
//     // Mg [theta][a][b][y][x]
//     // Og [theta][a][b][y][x]
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int idx = y * img_width + img_height;
//     printf("(%d, %d) voting on ellipses at all\n", y, x);
//     if (x > 0 && y > 0 && x < img_width - 1 && y < img_height - 1 && gradients[y*img_width + x] ) {
        
//         //pii p = {y,x};
//         pii p;
//         p.first = y;
//         p.second = x;
//         double dy = (gradY[idx] / gradients[idx]);
//         double dx = (gradX[idx] / gradients[idx]);

//         int size_a = sizeof(d_a_vals) / sizeof(int);
//         int size_b = sizeof(d_b_vals) / sizeof(int);
//         // printf("(%d, %d) -> {%f, %f} |%f|\n", i, j, gradY[i][j], gradX[i][j], gradients[i][j]);
//         // printf("processing pixel %d, %d\n", i, j);
//         // for(int degr = 0; degr < 360; degr += angular_granularity){
//         for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++) {
//             int degr = degr_idx * angular_granularity;
//             double theta = degr * PI/180.0;
//             for(int a_idx = 1; a_idx < size_a; a_idx++){
//                 for(int b_idx = 1; b_idx < size_b; b_idx++){
//                     int a = d_a_vals[a_idx];
//                     int b = d_b_vals[b_idx];
//                     double G[2][2] = {
//                         {a * cos(theta), -b * sin(theta)},
//                         {a * sin(theta),  b * cos(theta)},
//                     };
//                     double transform_matrix[2][2];
//                     M_mults(G, transform_matrix);
//                     pdd grad_t;
//                     grad_t.first = dy * transform_matrix[0][0] + dx * transform_matrix[0][1];
//                     grad_t.second = dy * transform_matrix[1][0] + dx * transform_matrix[1][1];
//                     // printf("generated transformed gradients {%f, %f} from {%f, %f}\n", grad_t.first, grad_t.second, dy, dx);
//                     for(int n = 1; n < MAX_RADII; n++){
//                         pdd p_plus;
//                         p_plus.first = p.first + grad_t.first * n;
//                         p_plus.second = p.second + grad_t.second * n;

//                         pdd p_minus;
//                         p_minus.first = p.first - grad_t.first * n;
//                         p_minus.second = p.second - grad_t.second * n;
//                         // printf("for (theta,a,b) = (%d,%d,%d), have points +(%d,%d) -(%d,%d)\n",
//                         //         degr, a, b, p_plus.first, p_plus.second, p_minus.first, p_minus.second);
//                         if(p_plus.first >= 0 && p_plus.first < img_height && p_plus.second >= 0 && p_plus.first < img_width){
//                             int idx = (degr_idx * (360/angular_granularity) * size_a * size_b * img_width) + (a_idx * size_a * size_b * img_width) + (b_idx * size_b * img_width) + (p_plus.first * img_width) + p_plus.second;
//                             printf("(%d, %d) voting for ellipses on (%f, %f) + (%f)\n", y, x, p_plus.first, p_plus.second, gradients[y * img_width + x]);
//                             // Og[idx] += 1;
//                             atomicAdd(&Og[idx], 1.0);
//                             // Mg[idx] += gradients[y * width + x];
//                             atomicAdd(&Mg[idx], gradients[y * img_width + x]);
//                         }
//                         if(p_minus.first >= 0 && p_minus.first < img_height && p_minus.second >= 0 && p_minus.first < img_width){
//                             int idx = (degr_idx * (360/angular_granularity) * size_a * size_b * img_width) + (a_idx * size_a * size_b * img_width) + (b_idx * size_b * img_width) + (p_minus.first * img_width) + p_minus.second;
//                             // Og[idx] -= 1;
//                             atomicAdd(&Og[idx], -1.0);
//                             // Mg[idx] -= gradients[i][j];
//                             atomicAdd(&Mg[idx], -gradients[y * img_width + x]);
//                         }
//                     }        
//                 }
//             }
//         }
//     }
// }


__global__ void ellipseResponseMapKernel(double *Mg, double *Og, double *gradX, double *gradY, double *gradients){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size_a = sizeof(d_a_vals) / sizeof(int);
    int size_b = sizeof(d_b_vals) / sizeof(int);
    int idx = y * img_width + x;
    if(y < img_height && y >= 0 && x < img_width && x >= 0){
        if(gradients[idx] != 0){

            double dy = gradY[idx] / gradients[idx];
            double dx = gradX[idx] / gradients[idx];
            for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++){
                int degr = degr_idx * angular_granularity;
                double theta = degr * PI / 180.0;
                for(int a_idx = 1; a_idx < size_a; a_idx++){
                    for(int b_idx = 1; b_idx < size_b; b_idx++){
                        // printf("(%d, %d) in (%d, %d, %d)\n", y, x, degr_idx, a_idx, b_idx);
                        int a = d_a_vals[a_idx];
                        int b = d_b_vals[b_idx];
                        double G[2][2] = {
                            {a * cos(theta), -b * sin(theta)},
                            {a * sin(theta),  b * cos(theta)},
                        };
                        double transform_matrix[2][2];
                        M_mults(G, transform_matrix);
                        double grad_t_y = dy * transform_matrix[0][0] + dx * transform_matrix[0][1];
                        double grad_t_x = dy * transform_matrix[1][0] + dx * transform_matrix[1][1];
                        // int idx5d = degr_idx * size_a * size_b * img_width * img_height +
                        //     a_idx * size_b * img_width * img_height +
                        //     b_idx * img_width * img_height +
                        //     y * img_height +
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
                                            p_plus_y * img_height +
                                            p_plus_x;
                                Mg[idx5d] += gradients[idx];
                            }
                            if(p_minus_y >= 0 && p_minus_y < img_height && p_minus_x >= 0 && p_minus_x < img_width){
                                int idx5d = degr_idx * size_a * size_b * img_width * img_height +
                                            a_idx * size_b * img_width * img_height +
                                            b_idx * img_width * img_height +
                                            p_minus_y * img_height +
                                            p_minus_x;
                                Mg[idx5d] -= gradients[idx];
                            }
                        }
                    }
                }
            }
        }
            
    }
}

__global__ void checkMgExists(double *Mg){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size_a = sizeof(d_a_vals) / sizeof(int);
    int size_b = sizeof(d_b_vals) / sizeof(int);
    int idx = y * img_width + x;
    if(y < img_height && y >= 0 && x < img_width && x >= 0){
        double maxval = 0.0;
        for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++){
            for(int a_idx = 0; a_idx < size_a; a_idx++){
                for(int b_idx = 0; b_idx < size_b; b_idx++){
                    int idx5d = degr_idx * size_a * size_b * img_width * img_height +
                                        a_idx * size_b * img_width * img_height +
                                        b_idx * img_width * img_height +
                                        y * img_height +
                                        x;
                    if(Mg[idx5d] > maxval) maxval = Mg[idx5d];
                }
            }
        }
        if(maxval != 0.0)
            printf("(%d,%d) has response %f\n", y, x, maxval);
    }
}

__global__void checkExists(double *S){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * img_width + x;
    if(y < img_height && y >= 0 && x < img_width && x >= 0){
        if(S[idx] != 0) 
            printf("(%d, %d)")
    }
}

//gaussian blur the 5d feature maps
__global__ void generate_S1(double *M, double *O, double *S, double *S_nms,
                double *S_flat, double *S_flat_nms){
    double gauss[3][3] = {
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
                                        y * img_height +
                                        x;
                            int idx2 =  degr_idx * size_a * size_b * img_width * img_height +
                                        a_idx * size_b * img_width * img_height +
                                        b_idx * img_width * img_height +
                                        (dy + y - 1) * img_height +
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
__global__ void generate_S2(double *M, double *O, double *S, double *S_nms,
                double *S_flat, double *S_flat_nms){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size_a = sizeof(d_a_vals) / sizeof(int);
    int size_b = sizeof(d_b_vals) / sizeof(int);
    for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++){
        for(int a_idx = 0; a_idx < size_a; a_idx++){
            for(int b_idx = 0; b_idx < size_b; b_idx++){
                if(y < (img_height - 1) && y > 0 && x < (img_width - 1) && x >= 1){
                    double maxNeighbor = 0.0;
                    for(int dy = 0; dy < 3; dy++){
                        for(int dx = 0; dx < 3; dx++){        
                            int idx2 =  degr_idx * size_a * size_b * img_width * img_height +
                                        a_idx * size_b * img_width * img_height +
                                        b_idx * img_width * img_height +
                                        (dy + y - 1) * img_height +
                                        (dx + x -1);
                            if(dy != 1 || dx != 1) maxNeighbor = max(maxNeighbor, S[idx2]);
                        }
                    }
                    int idx = degr_idx * size_a * size_b * img_width * img_height +
                                a_idx * size_b * img_width * img_height +
                                b_idx * img_width * img_height +
                                y * img_height +
                                x;
                    if(S[idx] < maxNeighbor) S_nms[idx] = 0;
                    else S_nms[idx] = S[idx];
                }
                        
            }
        }
    }
    double maxresponse = 0.0;
    for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++){
        for(int a_idx = 0; a_idx < size_a; a_idx++){
            for(int b_idx = 0; b_idx < size_b; b_idx++){
                int idx = degr_idx * size_a * size_b * img_width * img_height +
                                a_idx * size_b * img_width * img_height +
                                b_idx * img_width * img_height +
                                y * img_height +
                                x;
                maxresponse = max(maxresponse, S[idx]);
            }
        }
    }
    S_flat_nms[y * img_width + x] = maxresponse;
}

__global__ void generate_S3(double *M, double *O, double *S, double *S_nms,
                double *S_flat, double *S_flat_nms){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size_a = sizeof(d_a_vals) / sizeof(int);
    int size_b = sizeof(d_b_vals) / sizeof(int);
    double gauss[3][3] = {
        {1.0/16.0, 2.0/16.0, 1.0/16.0},
        {2.0/16.0, 4.0/16.0, 2.0/16.0},
        {1.0/16.0, 2.0/16.0, 1.0/16.0}
    };

    for(int dy = 0; dy < 3; dy++){
        for(int dx = 0; dx < 3; dx++){
            S_flat[y * img_width + x] += S_flat_nms[(dy+y-1)*img_width + dx+x-1] * gauss[dy][dx];
        }
    }
}

__global__ void generate_S4(double *M, double *O, double *S, double *S_nms,
                double *S_flat, double *S_flat_nms){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size_a = sizeof(d_a_vals) / sizeof(int);
    int size_b = sizeof(d_b_vals) / sizeof(int);
    double maxNeighbor = 0.0;
    for(int dy = 0; dy < 3; dy++){
        for(int dx = 0; dx < 3; dx++){
            if(dy != 1 || dx != 1) maxNeighbor = max(maxNeighbor, S_flat[(dy+y-1)*img_width + dx+x-1]);
        }
    }
    if(maxNeighbor > S_flat[y * img_width + x]) S_flat_nms[y * img_width + x] = 0;
    else S_flat_nms[y * img_width + x] = S_flat[y * img_width + x];
}

__global__ void grayscaleKernel(int *color_image_d, double *grayscaleImg) {
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

__global__ void gradientKernel(double *grayscaleImg, double *gradients, double *gradX, double *gradY, double *gradientDir) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int flat_idx = get_2d_idx(y, x);
    if (x > 0 && y > 0 && x < (img_width - 1) && y < (img_height - 1)) {
        double xSobel[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
        double ySobel[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
        double xg = 0, yg = 0;
        if(y == 504 && x == 590){
            printf("gray value is %f\n ", grayscaleImg[flat_idx]);
        }
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                double pixel = grayscaleImg[get_2d_idx(y + dy, x + dx)];
                xg += pixel * xSobel[dy + 1][dx + 1];
                yg += pixel * ySobel[dy + 1][dx + 1];
                if(y == 504 && x == 590){
                    printf("iterating through (%d, %d) gray =%f\n", y+dy, x+dx, pixel);
                }
                
            }
        }
        
        double mag = sqrt(xg * xg + yg * yg);
        gradients[flat_idx] = (mag > 200) ? mag : 0;
        if(y == 504 && x == 590){
            printf("(%d, %d) has grads, (%f, %f) = %f\n", y, x, xg, yg, mag);
        }
        gradX[flat_idx] = xg;
        gradY[flat_idx] = yg;
        double angle = atan(yg/xg);
        if(isnan(gradientDir[flat_idx])) gradientDir[flat_idx] = PI / 2 * ((yg > 0) - (yg < 0));
    }
    else {
        // printf("(%d, %d) (%d,%d) did not write to gradients\n", y, x, img_height, img_width);
    }
    // __syncthreads();
}

__global__ void NMSKernel(double *gradients, double *gradientDir, double *nms_gradients) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < img_width - 1 && y < img_height - 1) {
        int idx = y * img_width + x;
        double n1, n2;
        if (gradients[idx] != 0) {
            double dir = gradientDir[idx];
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

__global__ void radialSymmetryKernel(double *gradX, double *gradY, double *gradients, double *O, double *M) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < img_width && y < img_height && gradients[y * img_width + x] != 0) {
        double dy = gradY[y * img_width + x] / gradients[y * img_width + x];
        double dx = gradX[y * img_width + x] / gradients[y * img_width + x];

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

__global__ void gaussConvolve1(double *M, double *postGauss, int *radii){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    double gauss[3][3] = {
        {1.0/16.0, 2.0/16.0, 1.0/16.0},
        {2.0/16.0, 4.0/16.0, 2.0/16.0},
        {1.0/16.0, 2.0/16.0, 1.0/16.0}
    };

    double t = 0;
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


__global__ void gaussConvolve2(double *M, double *postGauss, int *radii){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    double gauss[3][3] = {
        {1.0/16.0, 2.0/16.0, 1.0/16.0},
        {2.0/16.0, 4.0/16.0, 2.0/16.0},
        {1.0/16.0, 2.0/16.0, 1.0/16.0}
    };
    double g = 0;
    for(int dy = 0; dy < 3; dy++){
        for(int dx = 0; dx < 3; dx++){
            g += M[(y + dy) * img_width + (x + dx)] * gauss[dy][dx];
        }
    }
    postGauss[y * img_width + x] = (g > THRESH) ? g : 0;
    // __syncthreads();
}

__global__ void postGaussNMSKernel(double *postGauss, double *gaussNMS, int *radii) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < img_width - 1 && y < img_height - 1) {
        int idx = y *img_width + x;
        double maxNeighbor = 0.0;
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

    unsigned long start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    //determine kernel call params
    dim3 blockDim(32,32, 1);
    dim3 gridDim((img_width_c + blockDim.x - 1) / blockDim.x,
                 (img_height_c + blockDim.y - 1) / blockDim.y,
                 1);
    // dim3 gridDim(1, 1, 1);


    //initialzie data structures
    double *grayscaleImage_d; int *color_img_d;
    int nBytesToCopy = img_width_c*img_height_c*3*sizeof(int);
    printf("transferring %d bytes\n", nBytesToCopy);
    cudaMalloc(&color_img_d, img_width_c*img_height_c*3*sizeof(int));
    cudaMemcpy(color_img_d, color_img_cpu, img_width_c*img_height_c*3*sizeof(int),  cudaMemcpyHostToDevice);
    cudaMalloc(&grayscaleImage_d, img_width_c * img_height_c * sizeof(double));

    double *gradients_d, *gradientDir_d, *gradX_d, *gradY_d, *nms_gradients_d;
    cudaMalloc(&gradients_d, img_width_c * img_height_c * sizeof(double));
    cudaMalloc(&gradientDir_d, img_width_c * img_height_c * sizeof(double));
    cudaMalloc(&gradX_d, img_width_c * img_height_c * sizeof(double));
    cudaMalloc(&gradY_d, img_width_c * img_height_c * sizeof(double));
    cudaMalloc(&nms_gradients_d, img_width_c * img_height_c * sizeof(double));

    double *O_d, *M_d;
    cudaMalloc(&O_d, MAX_RADII * img_width_c * img_height_c * sizeof(double));
    cudaMalloc(&M_d, MAX_RADII * img_width_c * img_height_c * sizeof(double));

    double *postGauss_d, *gaussNMS_d; int *radii_d;
    cudaMalloc(&postGauss_d, img_width_c * img_height_c * sizeof(double));
    cudaMalloc(&gaussNMS_d, img_width_c * img_height_c * sizeof(double));
    cudaMalloc(&radii_d,     img_width_c * img_height_c * sizeof(int));

    double *Mg_d, *Og_d, *S_d, *S_nms_d, *S_flat_d, *S_nms_flat_d;
    cudaMalloc(&Mg_d, img_width_c * img_height_c * (360/angular_granularity) * a_vals.size() * b_vals.size() * sizeof(double));
    cudaMalloc(&Og_d, img_width_c * img_height_c * (360/angular_granularity) * a_vals.size() * b_vals.size() * sizeof(double));
    cudaMalloc(&S_d, img_width_c * img_height_c * (360/angular_granularity) * a_vals.size() * b_vals.size() * sizeof(double));
    cudaMalloc(&S_nms_d, img_width_c * img_height_c * (360/angular_granularity) * a_vals.size() * b_vals.size() * sizeof(double));
    cudaMalloc(&S_flat_d, img_width_c * img_height_c * sizeof(double));
    cudaMalloc(&S_nms_flat_d, img_width_c * img_height_c * sizeof(double));

    //cpu mallocs
    double *postGauss_c, *gaussNMS_c, *S_flat_c, *S_flat_nms_c;
    int *radii_c;
    postGauss_c = (double*) malloc(img_width_c * img_height_c * sizeof(double));
    gaussNMS_c  = (double*) malloc(img_width_c * img_height_c * sizeof(double));
    radii_c     = (int*) malloc(img_width_c * img_height_c * sizeof(int));
    S_flat_c = (double*) malloc(img_width_c * img_height_c * sizeof(double));
    S_flat_nms_c = (double*) malloc(img_width_c * img_height_c * sizeof(double));


    //grayscale the image
    printf("grayscale start\n");
    grayscaleKernel<<<gridDim, blockDim>>>(color_img_d, grayscaleImage_d);
    cudaThreadSynchronize();
    printf("grayscale end\n");

    if(LOG){
        double *grayscale_c = (double*) malloc(img_height_c * img_width_c * sizeof(double));
        cudaMemcpy(grayscale_c, grayscaleImage_d, img_width_c * img_height_c * sizeof(double), cudaMemcpyDeviceToHost);
        ofstream gray_file("images_cuda/" + basefile + "_gray.txt");
        for(int i = 0; i < img_height_c; i++){
            for(int j = 0; j < img_width_c; j++){
                gray_file << grayscale_c[i * img_width_c + j] << endl;
            }
        }
        gray_file.close();
    }

    printf("gradient calculation start\n");
    gradientKernel<<<gridDim, blockDim>>>(grayscaleImage_d, gradients_d, gradX_d, gradY_d, gradientDir_d);
    cudaThreadSynchronize();
    printf("gradient calculation end\n");

    if(LOG){
        double *gradients_c = (double*) malloc(img_height_c * img_width_c * sizeof(double));
        cudaMemcpy(gradients_c, gradients_d, img_width_c * img_height_c * sizeof(double), cudaMemcpyDeviceToHost);
        ofstream grad_file("images_cuda/" + basefile + "_gradients.txt");
        for(int i = 0; i < img_height_c; i++){
            for(int j = 0; j < img_width_c; j++){
                grad_file << gradients_c[i * img_width_c + j] << endl;
            }
        }
        grad_file.close();
    }

    printf("NMS start\n");
    NMSKernel<<<gridDim,blockDim>>>(gradients_d, gradientDir_d, nms_gradients_d);
    cudaThreadSynchronize();
    printf("NMS end\n");

    printf("radial symmetry calculation start\n");
    radialSymmetryKernel<<<gridDim,blockDim>>>(gradX_d, gradY_d, gradients_d, O_d, M_d);
    cudaThreadSynchronize();
    printf("radial symmetry calculation end\n");

    printf("best response calculation start\n");
    gaussConvolve1<<<gridDim, blockDim>>>(M_d, postGauss_d, radii_d);
    cudaThreadSynchronize();
    printf("best response calculation end\n");

    printf("blur start\n");
    gaussConvolve2<<<gridDim, blockDim>>>(M_d, postGauss_d, radii_d);
    cudaThreadSynchronize();
    printf("blur end\n");

    cudaMemcpy(postGauss_c, postGauss_d, img_width_c * img_height_c * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(radii_c, radii_d, img_width_c * img_height_c * sizeof(int), cudaMemcpyDeviceToHost);

    printf("NMS on best-radii start\n");
    postGaussNMSKernel<<<gridDim, blockDim>>>(postGauss_d, gaussNMS_d, radii_d);
    cudaThreadSynchronize();
    printf("NMS on best-radii end\n");

    cudaMemcpy(gaussNMS_c, gaussNMS_d, img_width_c * img_height_c * sizeof(double), cudaMemcpyDeviceToHost);

    /*
    
        ELLIPSES FROM HERE, CIRCLES BEFORE

    */

    printf("ellipse response map start\n");
    // ellipseResponseMapKernel<<<gridDim,blockDim>>>(Mg_d, Og_d, gradX_d, gradY_d, nms_gradients_d, gradientDir_d);
    ellipseResponseMapKernel<<<gridDim, blockDim>>>(Mg_d, Og_d, gradX_d, gradY_d, nms_gradients_d);
    cudaThreadSynchronize();
    printf("ellipse resposne map end\n");

    // printf("checking maxes start\n");
    // checkMgExists<<<gridDim, blockDim>>>(Mg_d);
    // cudaThreadSynchronize();
    // printf("checking maxes end\n");




    printf("S1 start\n");
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
    printf("S4 start\n");
    generate_S4<<<gridDim, blockDim>>>(Mg_d, Og_d, S_d, S_nms_d, S_flat_d, S_nms_flat_d);
    cudaThreadSynchronize();
    printf("S4 end\n");



    


    // cudaMemcpy(postGauss_c, postGauss_d, img_width_c * img_height_c * sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(gaussNMS_c, gaussNMS_d, img_width_c * img_height_c * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(radii_c, radii_d, img_width_c * img_height_c * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(S_flat_c, S_flat_d, img_width_c * img_height_c * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(S_flat_nms_c, S_nms_flat_d, img_width_c * img_height_c * sizeof(double), cudaMemcpyDeviceToHost);

    unsigned long end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

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
                if(S_flat_c[i * img_width_c + j] != 0) printf("(%d,%d)\n", i, j);
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
    return end - start;
}


int main(){
    unsigned long runtime = driverCuda();
    printf("Runtime: %ld\n", runtime);
}