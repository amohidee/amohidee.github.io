#include <cstdint>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <wand/MagickWand.h>

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

MagickWandGenesis();
magick_wand=NewMagickWand();
status=MagickReadImage(magick_wand,argv[1]);
if (status == MagickFalse)
ThrowWandException(magick_wand);
/*
Turn the images into a thumbnail sequence.
*/
MagickResetIterator(magick_wand);
while (MagickNextImage(magick_wand) != MagickFalse)
MagickResizeImage(magick_wand,106,80,LanczosFilter,1.0);
/*
Write the image then destroy it.
*/
status=MagickWriteImages(magick_wand,argv[2],MagickTrue);
if (status == MagickFalse)
ThrowWandException(magick_wand);
magick_wand=DestroyMagickWand(magick_wand);
MagickWandTerminus();
return(0);

for (y=0; y < (long) MagickGetImageHeight(image_wand); y++)
  {
    pixels=PixelGetNextIteratorRow(iterator,&width);
    contrast_pixels=PixelGetNextIteratorRow(contrast_iterator,&width);
    if ((pixels == (PixelWand **) NULL) ||
        (contrast_pixels == (PixelWand **) NULL))
      break;
    for (x=0; x < (long) width; x++)
    {

     // do stuff here
        red = PixelGetRed(pixels[x]);
        green = PixelGetGreen(pixels[x]);
        blue = PixelGetBlue(pixels[x]);

        Pixel newPixel = malloc(sizeof(Pixel));
        newPixel.r = red;
        

      
    }
    (void) PixelSyncIterator(contrast_iterator);
  }


__device__ __inline__ double convolve(double* a, double* b, int size){
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
    grayscaleImage[pixelY][pixelX] = .2125f * colorImage->img->r +
                                     .7154f * colorImage->img->g + 
                                     .0721f * colorImage->img->b;
}

__global__ void gradientKernel(float* gradientArr, float* grayscaleImage, int w, int h){
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

    grayscaleKernel<<<gridDim, blockDim>>>(grayscaleImage, colorImage)
    gradientKernel<<<gridDim, blockDim>>>(gradientArr, grayscaleImage, colorImage->w, colorImage->h);
}