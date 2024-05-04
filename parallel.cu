#include <bits/stdc++.h>
using namespace std;

#define PI 3.14159265358979323846
#define MAX_RADII 50
#define Kn 9.9
#define THRESH 1000

typedef pair<int,int> pii;

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


__global__ void grayscaleKernel(Pixel* img, double* grayscaleImg, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        Pixel pixel = img[idx];
        grayscaleImg[idx] = 0.2125 * pixel.r + 0.7154 * pixel.g + 0.0721 * pixel.b;
    }
}


void gradient(double** grayscaleImg, double** gradients, double** gradientDir, Image* color,
              double** gradX, double** gradY){
    double xSobel[3][3] = {
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1},
    };
    double ySobel[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1},
    };

    for(int i = 1; i < color->h - 1; i++){
        for(int j = 1; j < color->w - 1; j++){
            // printf("%d, %d\n", i, j);
            double xg, yg;
            xg = 0;
            yg = 0;
            for(int dy = 0; dy < 3; dy++){
                for(int dx = 0; dx < 3; dx++){
                    xg += grayscaleImg[dy+i-1][dx+j-1] * xSobel[dy][dx];
                    yg += grayscaleImg[dy+i-1][dx+j-1] * ySobel[dy][dx];
                }
            }
            double mag = sqrt(xg*xg + yg*yg);
            gradients[i][j] = (mag > 200) ? mag : 0;
            gradY[i][j] = yg; gradX[i][j] = xg;
// ")
            // if(gradients[i][j]){
            //     gradientDir[i][j] = atan(yg/xg);
            //     if(isnan(gradientDir[i][j])) gradientDir[i][j] = PI / 2 * ((yg > 0) - (yg < 0));
            // }
            // else {
            //     gradientDir[i][j] = -10;
            // }
            gradientDir[i][j] = atan(yg/xg);
            if(isnan(gradientDir[i][j])) gradientDir[i][j] = PI / 2 * ((yg > 0) - (yg < 0));
        }
    }
}

__global__ void gradientKernel(double** grayscaleImg, double** gradients, double** gradX, double** gradY, double** gradientDir, int width, int height) {
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
        gradients[y][x] = (mag > 200) ? mag : 0;
        gradX[y][x] = xg;
        gradY[y][x] = yg;
        double angle = atan2(yg, xg);
        if(isnan(gradientDir[y][x])) gradientDir[y][x] = PI / 2 * ((yg > 0) - (yg < 0));
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
        if(isnan(gradientDir[y * width + x])) gradientDir[y * width + x] = PI / 2 * ((yg > 0) - (yg < 0));
    }
}

__global__ void NMSKernel(double **gradients, double **gradientDir, double **nms_gradients, int width, int height, double PI) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int idx = y * width + x;
        double n1, n2;
        if (gradients[y][x] != 0) {
            double dir = gradientDir[y][x];
            if (dir > PI / 4 || dir < -PI / 4) {
                n1 = gradients[y - 1][x];
                n2 = gradients[y + 1][x];
            } else if (dir > 0) {
                n1 = gradients[y - 1][x + 1];
                n2 = gradients[y - 1][x - 1];
            } else if (dir > -PI / 4) {
                n1 = gradients[y + 1][x + 1];
                n2 = gradients[y - 1][x - 1];
            } else {
                n1 = 0;
                n2 = 0;
            }

            if (gradients[y][x] >= n1 && gradients[y][x] >= n2) {
                nms_gradients[y][x] = gradients[y][x];
            } else {
                nms_gradients[y][x] = 0;
            }
        }
    }
}

__global__ void NMSKernel(double *gradients, double *gradientDir, double *nms_gradients, int width, int height, double PI) {
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

            if (p_plus_y >= 0 && p_plus_y < height && p_plus_x >= 0 && p_plus_x < width) {
                atomicAdd(&O[r * height * width + p_plus_y * width + p_plus_x], 1);
                atomicAdd(&M[r * height * width + p_plus_y * width + p_plus_x], gradients[y * width + x]);
            }
            if (p_minus_y >= 0 && p_minus_y < height && p_minus_x >= 0 && p_minus_x < width) {
                atomicAdd(&O[r * height * width + p_minus_y * width + p_minus_x], -1);
                atomicAdd(&M[r * height * width + p_minus_y * width + p_minus_x], -gradients[y * width + x]);
            }
        }
    }
}


//gaussian convolve
__global__ void gaussConvolve(double ***M, double **postGauss, int **radii, Image *color){
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

    for(int i = 0; i < color->h; i++){
        for(int j = 0; j < color->w; j++){
            double t = 0;
            int best_r = 0;
            for(int r = 0; r < MAX_RADII; r++){
                // t = max(M[r][i][j], t);
                if(M[r][i][j] > t){
                    best_r = r;
                    t = M[r][i][j];
                }
            }
            radii[i][j] = best_r;
            M[0][i][j] = t;
        }
    }
    for(int i = 1; i < color->h - 1; i++){
        for(int j = 1; j < color->w - 1; j++){
            // printf("(%d, %d)", i, j);
            double g = 0;
            for(int dy = 0; dy < 3; dy++){
                for(int dx = 0; dx < 3; dx++){
                    // printf("(%d, %d), %f || ", dy, dx, gauss[dy][dx]);
                    g += M[0][dy+i-1][dx+j-1] * gauss[dy][dx];
                    if(g != 0){
                        // printf("(%d,%d) %f, %f\n", i, j, M[0][dy+i-1][dx+j-1], gauss[dy][dx]);
                    }
                }
            }
            // printf("\n");
            // printf("(%d,%d)->%f\n", i,j, g);
            postGauss[i][j] = (g > THRESH) ? g : 0;
        }
    }
}

void postGaussNMS(double **postGauss, double **gaussNMS, int **radii,Image* color){
    for(int i = 1; i < color->h -1; i++){
        for(int j = 1; j < color->w -1; j++){
            double maxNeighbor = 0.0;
            for(int dy = -1; dy < 2; dy++){
                for(int dx = -1; dx < 2; dx++){
                    if(dy != 0 || dx != 0){
                        if(maxNeighbor < postGauss[i+dy][j+dx]){
                            maxNeighbor = postGauss[i+dy][j+dx];
                        }
                        // maxNeighbor = max(maxNeighbor, postGauss[i + dy][j + dx], maxNeighbor);
                    }
                }
            }
            if(postGauss[i][j] >= maxNeighbor){
                gaussNMS[i][j] = postGauss[i][j];
            }
            else {
                radii[i][j] = 0;
            }
        }
    }
}