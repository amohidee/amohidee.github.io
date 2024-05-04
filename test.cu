#include <cstdint>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <wand/MagickWand.h>
#include <stdio.h>


#define PI 3.14159265358979323846

long
    y;

MagickBooleanType
    status;

MagickWand
    *contrast_wand,
    *image_wand;

PixelInfo
    pixel;

PixelIterator
    *contrast_iterator,
    *iterator;

PixelWand
    **contrast_pixels,
    **pixels;

register long
x;

unsigned long
width;

typedef struct Pixel {
    int8_t r,g,b;
} Pixel;

typedef struct Image {
    int w, h;
    Pixel** img;
    Image(int width, int height) {
        w = width; 
        h = height;
        cudaMalloc(img, sizeof(Pixel*) * h);
        for(int i = 0; i < h ;i++){
            cudaMalloc(img[i], sizeof(Pixel) * w);
        }
    }
} Image;

void multiplyMatrices(int A[2][2], int B[2][2], int result[2][2]) {
    // Initialize result
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            result[i][j] = 0;
        }
    }

    // parallelize?
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void getRotationMatrix(float theta, float** R) {
    R[0][0] = cos(theta);
    R[0][1] = -sin(theta);
    R[1][0] = sin(theta);
    R[1][1] = cos(theta);
}

void getScalingMatrix(float a, float b, float** S) {
    S[0][0] = a;
    S[0][1] = 0;
    S[1][0] = 0;
    S[1][1] = b;
}


__device__ __inline__ double convolve(double** a, double** b, int size){
    double r = 0;
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            r += a[i][j] * b[i][j];
        }
    }
    return r;
}

__global__ void grayscaleKernel(float* grayscaleImage, Image* colorImage) {
    int pixelX = (blockIdx.x * blockDim.x) + threadIdx.x;
    int pixelY = (blockIdx.y * blockDim.y) + threadIdx.y;
    if(pixelX < 0 || pixelX >= colorImage.w || pixelY < 0 || pixelY >= colorImage.h)
        return;
    grayscaleImage[pixelY][pixelX] = .2125f * colorImage->img[pixelY][pixelX].r +
                                     .7154f * colorImage->img[pixelY][pixelX].g + 
                                     .0721f * colorImage->img[pixelY][pixelX].b;
}

__global__ void gradientKernel(float* gradientArr, float* gradientMagnitude, float* grayscaleImage, int w, int h){
    int pixelX = (blockIdx.x * blockDim.x) + threadIdx.x;
    int pixelY = (blockIdx.y * blockDim.y) + threadIdx.y;

    int localPixels[3][3];

    if(pixelX == 0) {
        for(int i = 0; i <3 ; i++){
            localPixels[0][i] = 0;
        }
    }

    if(pixelY == 0) {
        for(int i = 0; i <3 ; i++){
            localPixels[i][0] = 0;
        }
    }
    if(pixelX == w-1) {
        for(int i = 0; i <3 ; i++){
            localPixels[2][i] = 0;
        }
    }
    if(pixelY == h-1) {
        for(int i = 0; i <3 ; i++){
            localPixels[i][2] = 0;
        }
    }

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            int lclX = pixelX + i -1;
            int lclY = pixelY + j -1;
            if(lclX < w && lclY < h && lclX >= 0 && lclY >= 0){
                localPixels[j][i] = grayscaleImage[lclY][lclX];
            } 
        }
    }
    double xSobel[3][3] = {
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1},
    }
    double ySobel[3][3] = {
        {1, 2, -1},
        {0, 0, 0},
        {-1, -2, -1},
    }
    double xGrad = convolve(localPixels, xSobel, 3);
    double yGrad = convolve(localPixels, ySobel, 3);
    gradientArr[pixelY][pixelX] = atan(xGrad, yGrad);
    gradientMagnitude[pixelY][pixelX] = sqrt(xGrad*xGrad + yGrad*yGrad)
}

__device__ void nmsKernel(float** gradientArr, float *gradientMagnitude, float *nmsGradient,int w, int h){
    int pixelX = (blockIdx.x * blockDim.x) + threadIdx.x;
    int pixelY = (blockIdx.y * blockDim.y) + threadIdx.y;

    if(pixelX < 1 || pixelX >=(w-1) || pixelY < 1 || pixelY >- (h-1))
        return;
    float angleDegrees = 180 / PI * gradientArr[pixelY][pixelX];
    int direction = (int)(angleDegrees) / 45;

    float n1, n2;

    if(direction == -1){
        /*
        \
         \
          \       
        */
        n1 = gradientMagnitude[pixelY + 1][pixelX - 1];
        n2 = gradientMagnitude[pixelY - 1][pixelX + 1];
    }
    else if(direction == 0){
        n1 = gradientMagnitude[pixelY + 1][pixelX];
        n2 = gradientMagnitude[pixelY - 1][pixelX];
    }
    else if(direction == 1){
        n1 = gradientMagnitude[pixelY + 1][pixelX + 1];
        n2 = gradientMagnitude[pixelY - 1][pixelX - 1];
    }
    else {
        n1 = 0; n2 = 0;
    }
    if(gradientMagnitude[pixelY][pixelX] > n1 && gradientMagnitude[pixelY][pixelX] > n2){
        nmsGradient[pixelY][pixelX] = gradientMagnitude[pixelY][pixelX];
    }
}

/*
    need to handle cpu -> gpu && gpu -> cpu outside this
    colorImage, device pointer to a color image struct
    gradientArr, device pointer to a 2d double array (w x h)

    cudaMalloc(gradientArr, colorImage.w * sizeof(gradientArr));
    for(int i = 0; i < colorImage->h; i++) cudaMalloc(gradientArr[i], sizeof(double) * colorImage->w);
*/
void calculateImageGradient(Image* colorImage, double** gradientArr) {
    dim3 blockDim(32,32,1);
    dim3 gridDim((colorImage->w + blockDim.x - 1) / blockDim.x, 
                 (colorImage->h + blockDim.y - 1) / blockDim.y,
                 1);
    //technically a temporary? maybe use for debugging need to move to arguments
    double** grayscaleImage;
    cudaMalloc(grayscaleImage, colorImage.w * sizeof(grayscaleImage));
    for(int i = 0; i < colorImage->h; i++) cudaMalloc(grayscaleImage[i], sizeof(double) * colorImage->w);

    grayscaleKernel<<<gridDim, blockDim>>>(grayscaleImage, colorImage);
    gradientKernel<<<gridDim, blockDim>>>(gradientArr, grayscaleImage, colorImage->w, colorImage->h);
}



void gradientVoting(float* gradientMagnitude, float* gradientDirection,
                    float* orientationProjection, float* magnitudeProjection,
                    int rows, int cols, int radius) {



    // radius is set beforehand, parallelize across all radii?
    // this is for FRST need to chaage to GFRST
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                int index = y * cols + x;
                float g_mag = gradientMagnitude[index];
                float g_dir = gradientDirection[index];




            }
        }


    }


void gradientVotingGFRST(float** grad_x, float** grad_y, float** orientationProjection, float** magnitudeProjection, 
                         int rows, int cols, int radius, float theta, float a, float b) {
    Matrix2x2 G = matrixMultiply(getRotationMatrix(theta), getScalingMatrix(a, b));
    // Assuming an appropriate matrix M and its inverse is predefined or calculated elsewhere

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float transformed_gx, transformed_gy;
            transformGradient(grad_x[y][x], grad_y[y][x], &transformed_gx, &transformed_gy, G);

            int pos_x = x + (int)(radius * transformed_gx);
            int pos_y = y + (int)(radius * transformed_gy);
            int neg_x = x - (int)(radius * transformed_gx);
            int neg_y = y - (int)(radius * transformed_gy);

            // Vote for symmetry
            if (pos_x >= 0 && pos_x < cols && pos_y >= 0 && pos_y < rows) {
                orientationProjection[pos_y][pos_x] += 1;
                magnitudeProjection[pos_y][pos_x] += sqrt(grad_x[y][x] * grad_x[y][x] + grad_y[y][x] * grad_y[y][x]);
            }
            if (neg_x >= 0 && neg_x < cols && neg_y >= 0 && neg_y < rows) {
                orientationProjection[neg_y][neg_x] -= 1;
                magnitudeProjection[neg_y][neg_x] -= sqrt(grad_x[y][x] * grad_x[y][x] + grad_y[y][x] * grad_y[y][x]);
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>\n";
        return EXIT_FAILURE;
    }

    MagickWandGenesis();
    MagickWand* image_wand = NewMagickWand();

    // Load the image
    if (MagickReadImage(image_wand, argv[1]) == MagickFalse) {
        ThrowWandException(image_wand);
        return EXIT_FAILURE;
    }

    // Get image dimensions
    unsigned long width = MagickGetImageWidth(image_wand);
    unsigned long height = MagickGetImageHeight(image_wand);

    // Create an Image 
    Image* hostImage = new Image(width, height);

    for (y=0; y < height; y++) {

        pixels=PixelGetNextIteratorRow(iterator,&width);
        contrast_pixels=PixelGetNextIteratorRow(contrast_iterator,&width);
        if ((pixels == (PixelWand **) NULL))
            break;
        for (x=0; x < width; x++)
            {

            // do stuff here
            red = PixelGetRed(pixels[x]);
            green = PixelGetGreen(pixels[x]);
            blue = PixelGetBlue(pixels[x]);

            Pixel newPixel = malloc(sizeof(Pixel));
            newPixel.r = red;
            newPixel.g= green;
            newPixel.b = blue;
            
            hostImage[y * width + x] = newPixel;

            
            }
        (void) PixelSyncIterator(contrast_iterator);
    }

    



    // Allocate GPU memory for the image
    Image* deviceImage;
    cudaMalloc(&deviceImage, sizeof(Image));
    cudaMemcpy(deviceImage, hostImage, sizeof(Image), cudaMemcpyHostToDevice);

    // Allocate memory for the gradient array
    double** hostGradientArray;
    double** deviceGradientArray;
    hostGradientArray = new double*[height];
    cudaMalloc(&deviceGradientArray, height * sizeof(double*));

    for (int i = 0; i < height; i++) {
        cudaMalloc(&(deviceGradientArray[i]), width * sizeof(double));
    }

    // allocate gradientMagnitude

    // Calculate the gradient
    calculateImageGradient(deviceImage, deviceGradientArray);

    // Copy the gradient back to host
    for (int i = 0; i < height; i++) {
        hostGradientArray[i] = new double[width];
        cudaMemcpy(hostGradientArray[i], deviceGradientArray[i], width * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // TODO: save the gradient to an output file 
    FILE* file = fopen(argv[2], "w");  // Open the file for writing
    if (file == NULL) {
        perror("Failed to open file");
        return;
    }

    int len = sizeof(hostGradientArray) / sizeof(hostGradientArray[0]);
    for (size_t i = 0; i < len; i++) {
        fprintf(file, "%d\n", hostGradientArray[i]);  // Write each array element to the file
    }

    fclose(file);  // Close the file

    DestroyMagickWand(image_wand);
    MagickWandTerminus();

    // Free memory
    cudaFree(deviceImage);
    for (int i = 0; i < height; i++) {
        cudaFree(deviceGradientArray[i]);
        delete[] hostGradientArray[i];
    }
    delete[] hostGradientArray;
    cudaFree(deviceGradientArray);

    return EXIT_SUCCESS;
}