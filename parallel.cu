#include <bits/stdc++.h>
using namespace std;

#define PI 3.14159265358979323846
#define MAX_RADII 50
#define Kn 9.9
#define THRESH 1000
#define angular_granularity 60

//typedef pair<int,int> pii;

typedef struct pii {
    int first;
    int second;
} pii;

typedef struct pdd {
    double first;
    double second;
} pdd;



typedef struct Pixel {
    uint8_t r,g,b;
} Pixel;

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



__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void NMSKernel(double *gradients, double *gradientDir, double *nms_gradients, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int idx = y * width + x;
        double n1, n2;
        if (gradients[idx] != 0) {
            double dir = gradientDir[idx];
            if (dir > PI / 4 || dir < -PI / 4) {
                n1 = gradients[(y - 1) * width + x];
                n2 = gradients[(y + 1) * width + x];
            } else if (dir > 0) {
                n1 = gradients[(y - 1) * width + (x + 1)];
                n2 = gradients[(y + 1) * width + (x - 1)];
            } else if (dir > -PI / 4) {
                n1 = gradients[(y + 1) * width + (x + 1)];
                n2 = gradients[(y - 1) * width + (x - 1)];
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
}

__global__ void grayscaleKernel(Pixel* img, double* grayscaleImg, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        Pixel pixel = img[idx];
        grayscaleImg[idx] = 0.2125 * pixel.r + 0.7154 * pixel.g + 0.0721 * pixel.b;
    }
}


__global__ void gradientKernel(double* grayscaleImg, double* gradients, double* gradX, double* gradY, double* gradientDir, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        double xSobel[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
        double ySobel[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
        double xg = 0, yg = 0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                double pixel = grayscaleImg[(y + dy) * width + (x + dx)];
                xg += pixel * xSobel[dy + 1][dx + 1];
                yg += pixel * ySobel[dy + 1][dx + 1];
            }
        }
        double mag = sqrt(xg * xg + yg * yg);
        gradients[y * width + x] = (mag > 200) ? mag : 0;
        gradX[y * width + x] = xg;
        gradY[y * width + x] = yg;
        double angle = atan2(yg, xg);
        if(isnan(gradientDir[y * width + x])) {
            gradientDir[y * width + x] = PI / 2 * ((yg > 0) - (yg < 0));
        }
    }
}


__global__ void radialSymmetryKernel(double *gradX, double *gradY, double *gradients, double *O, double *M, int width, int height, int max_radii) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height && gradients[y * width + x] != 0) {
        double dy = gradY[y * width + x] / gradients[y * width + x];
        double dx = gradX[y * width + x] / gradients[y * width + x];

        for (int r = 1; r < max_radii; r++) {
            int p_plus_y = y + dy * r;
            int p_plus_x = x + dx * r;
            int p_minus_y = y - dy * r;
            int p_minus_x = x - dx * r;


            // jsut using atomic add but we can change it to the blocking method
            if (p_plus_y >= 0 && p_plus_y < height && p_plus_x >= 0 && p_plus_x < width) {
                atomicAddDouble(&O[r * height * width + p_plus_y * width + p_plus_x], 1.0);
                atomicAddDouble(&M[r * height * width + p_plus_y * width + p_plus_x], gradients[y * width + x]);
            }
            if (p_minus_y >= 0 && p_minus_y < height && p_minus_x >= 0 && p_minus_x < width) {
                atomicAddDouble(&O[r * height * width + p_minus_y * width + p_minus_x], -1.0);
                atomicAddDouble(&M[r * height * width + p_minus_y * width + p_minus_x], -gradients[y * width + x]);
            }
        }
    }
}


//gaussian convolve
__global__ void gaussConvolve1(double *M, double *postGauss, int *radii, Image *color, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // for(int i = 0; i < 3; i++){
    //         for(int j = 0; j < 3; j++){
    //             gauss[i][j] =  1/16 * gauss[i][j];
    //     }
    // }

    double t = 0;
    int best_r = 0;
    for(int r = 0; r < MAX_RADII; r++){
        // t = max(M[r][i][j], t);
        if(M[r*width*height + y*width + x] > t){
            best_r = r;
            t = M[r*width*height + y*width + x];
        }
    }
    radii[y*width + x] = best_r;
    M[0*width*height + y*width + x] = t;
}


__global__ void gaussConvolve2(double *M, double *postGauss, int *radii, Image *color, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    double gauss[3][3] = {
        {1.0/16.0, 2.0/16.0, 1.0/16.0},
        {2.0/16.0, 4.0/16.0, 2.0/16.0},
        {1.0/16.0, 2.0/16.0, 1.0/16.0}
    };

    // printf("(%d, %d)", i, j);
    double g = 0;
    for(int dy = 0; dy < 3; dy++){
        for(int dx = 0; dx < 3; dx++){
            // printf("(%d, %d), %f || ", dy, dx, gauss[dy][dx]);
            g += M[(y + dy) * width + (x + dx)] * gauss[dy][dx];
            if(g != 0){
                // printf("(%d,%d) %f, %f\n", i, j, M[0][dy+i-1][dx+j-1], gauss[dy][dx]);
            }
        }
    }
    // printf("\n");
    // printf("(%d,%d)->%f\n", i,j, g);
    postGauss[y * width + x] = (g > THRESH) ? g : 0;



}


__global__ void postGaussNMSKernel(double *postGauss, double *gaussNMS, int *radii, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int idx = y * width + x;
        double maxNeighbor = 0.0;
        // fidn the max value in the block and keep track of it
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int neighbor_idx = (y + dy) * width + (x + dx);
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
}

__global__ void ellipseResponseMapKernel(double *Mg, double *Og, double *gradX, double *gradY, double *gradients, double *gradientDir, Image *color, int* d_a_vals, int* d_b_vals, int width, int height){
    // Mg [theta][a][b][y][x]
    // Og [theta][a][b][y][x]
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + height;
    if (x > 0 && y > 0 && x < width - 1 && y < height - 1 && gradients[y*width + x] ) {
        
        //pii p = {y,x};
        pii p;
        p.first = y;
        p.second = x;
        double dy = (gradY[idx] / gradients[idx]);
        double dx = (gradX[idx] / gradients[idx]);

        int size_a = sizeof(d_a_vals) / sizeof(int);
        int size_b = sizeof(d_b_vals) / sizeof(int);
        // printf("(%d, %d) -> {%f, %f} |%f|\n", i, j, gradY[i][j], gradX[i][j], gradients[i][j]);
        // printf("processing pixel %d, %d\n", i, j);
        // for(int degr = 0; degr < 360; degr += angular_granularity){
        for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++) {
            int degr = degr_idx * angular_granularity;
            double theta = degr * PI/180.0;
            for(int a_idx = 1; a_idx < size_a; a_idx++){
                for(int b_idx = 1; b_idx < size_b; b_idx++){
                    int a = d_a_vals[a_idx];
                    int b = d_b_vals[b_idx];
                    double G[2][2] = {
                        {a * cos(theta), -b * sin(theta)},
                        {a * sin(theta),  b * cos(theta)},
                    };
                    // printf("generated G matrix = [ [%f, %f], [%f, %f]]\n", G[0][0], G[0][1], G[1][0], G[1][1]);
                    double transform_matrix[2][2];
                    M_mults(G, transform_matrix);
                    // printf("(%f,%d,%d)generated T matrix = [ [%f, %f], [%f, %f]]\n", 
                            // theta, a, b, transform_matrix[0][0], transform_matrix[0][1], transform_matrix[1][0], transform_matrix[1][1]);

                    // pair<double, double> grad_t = {
                    //     dy * transform_matrix[0][0] + dx * transform_matrix[0][1],
                    //     dy * transform_matrix[1][0] + dx * transform_matrix[1][1]
                    // };
                    pdd grad_t;
                    grad_t.first = dy * transform_matrix[0][0] + dx * transform_matrix[0][1];
                    grad_t.second = dy * transform_matrix[1][0] + dx * transform_matrix[1][1];
                    // printf("generated transformed gradients {%f, %f} from {%f, %f}\n", grad_t.first, grad_t.second, dy, dx);
                    for(int n = 1; n < MAX_RADII; n++){
                        pdd p_plus;
                        p_plus.first = p.first + grad_t.first * n;
                        p_plus.second = p.second + grad_t.second * n;

                        pdd p_minus;
                        p_minus.first = p.first - grad_t.first * n;
                        p_minus.second = p.second - grad_t.second * n;
                        // printf("for (theta,a,b) = (%d,%d,%d), have points +(%d,%d) -(%d,%d)\n",
                        //         degr, a, b, p_plus.first, p_plus.second, p_minus.first, p_minus.second);
                        if(p_plus.first >= 0 && p_plus.first < color->h && p_plus.second >= 0 && p_plus.first < color->w){
                            int idx = (degr_idx * (360/angular_granularity) * MAX_RADII * MAX_RADII * width) + (a_idx * MAX_RADII * MAX_RADII * width) + (b_idx * MAX_RADII * width) + (p_plus.first * width) + p_plus.second;
                            // Og[idx] += 1;
                            atomicAddDouble(&Og[idx], 1.0);
                            // Mg[idx] += gradients[y * width + x];
                            atomicAddDouble(&Mg[idx], gradients[y * width + x]);
                        }
                        if(p_minus.first >= 0 && p_minus.first < color->h && p_minus.second >= 0 && p_minus.first < color->w){
                            int idx = (degr_idx * (360/angular_granularity) * MAX_RADII * MAX_RADII * width) + (a_idx * MAX_RADII * MAX_RADII * width) + (b_idx * MAX_RADII * width) + (p_minus.first * width) + p_minus.second;
                            // Og[idx] -= 1;
                            atomicAddDouble(&Og[idx], -1.0);
                            // Mg[idx] -= gradients[i][j];
                            atomicAddDouble(&Mg[idx], -gradients[y * width + x]);
                        }
                    }
                    
                }
            }
        }

    }

}

/*


//gaussian convolve
__global__ void gaussConvolve1Ellipse(double ***M, double **postGauss, int **radii, Image *color, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    double gauss[3][3] = {
        {1.0/16.0, 2.0/16.0, 1.0/16.0},
        {2.0/16.0, 4.0/16.0, 2.0/16.0},
        {1.0/16.0, 2.0/16.0, 1.0/16.0}
    };

    // for(int i = 0; i < 3; i++){
    //         for(int j = 0; j < 3; j++){
    //             gauss[i][j] =  1/16 * gauss[i][j];
    //     }
    // }

    double t = 0;
    int best_r = 0;
    for(int r = 0; r < MAX_RADII; r++){
        // t = max(M[r][i][j], t);
        if(M[r*width*height + y*width + x] > t){
            best_r = r;
            t = M[r*width*height + y*width + x];
        }
    }
    radii[y*width + x] = best_r;
    M[0*width*height + y*width + x] = t;
}


__global__ void gaussConvolve2Ellipse(double *M, double *postGauss, int *radii, Image *color, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    double gauss[3][3] = {
        {1.0/16.0, 2.0/16.0, 1.0/16.0},
        {2.0/16.0, 4.0/16.0, 2.0/16.0},
        {1.0/16.0, 2.0/16.0, 1.0/16.0}
    };

    // printf("(%d, %d)", i, j);
    double g = 0;
    for(int dy = 0; dy < 3; dy++){
        for(int dx = 0; dx < 3; dx++){
            // printf("(%d, %d), %f || ", dy, dx, gauss[dy][dx]);
            g += M[(y + dy) * width + (x + dx)] * gauss[dy][dx];
            if(g != 0){
                // printf("(%d,%d) %f, %f\n", i, j, M[0][dy+i-1][dx+j-1], gauss[dy][dx]);
            }
        }
    }
    // printf("\n");
    // printf("(%d,%d)->%f\n", i,j, g);
    postGauss[y * width + x] = (g > THRESH) ? g : 0;



}

*/

int main() {
    const int a_vals[] = {2, 4, 6, 8};
    const int b_vals[] = {2, 4, 6, 8};
    int size_a = sizeof(a_vals) / sizeof(a_vals[0]);
    int size_b = sizeof(b_vals) / sizeof(b_vals[0]);


    int *d_a_vals, *d_b_vals;

    // Allocate memory on the device
    cudaMalloc((void**)&d_a_vals, size_a * sizeof(int));
    cudaMalloc((void**)&d_b_vals, size_b * sizeof(int));



    // Copy data from host to device
    cudaMemcpy(d_a_vals, a_vals, size_a * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_vals, b_vals, size_b * sizeof(int), cudaMemcpyHostToDevice);

}