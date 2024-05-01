#include <bits/stdc++.h>
using namespace std;

#define PI 3.14159265358979323846
#define MAX_RADII 20
#define Kn 9.9

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

void grayscale(Image* color, double** grayscaleImg){
    for(int i = 0; i < color->h; i++){
        for(int j = 0; j < color->w; j++){
            grayscaleImg[i][j] =    .2125f * color->img[i][j].r +
                                    .7154f * color->img[i][j].g + 
                                    .0721f * color->img[i][j].b;
        }
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
            gradients[i][j] = sqrt(xg*xg + yg*yg);
            gradY[i][j] = yg; gradX[i][j] = xg;
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

void NMS(double **gradients, double **gradientDir,double **nms_gradients,
            Image *color){
    for(int i = 1; i < color->h - 1; i++){
        for(int j = 1; j < color->w - 1; j++){
            float n1, n2;
            if(gradients[i][j] != 0) { //existent gradient
                if(gradientDir[i][j] > PI / 4 || gradientDir[i][j] < PI/(-4)) {
                    n1 = gradients[i-1][j];
                    n2 = gradients[i+1][j];
                }
                else if(gradientDir[i][j] > 0){
                    n1 = gradients[i-1][j+1];
                    n2 = gradients[i+1][j-1];
                }
                else if(gradientDir[i][j] > PI/(-4)){
                    n1 = gradients[i+1][j+1];
                    n2 = gradients[i-1][j-1];
                }
                else{
                    n1 = 0; n2 = 0;
                }

                if(gradients[i][j] >= n1 && gradients[i][j] >= n2){
                    nms_gradients[i][j] = gradients[i][j];
                }
            }
        }
    }
}

//circles
void radialSymmetry(double **gradX, double** gradY, double **gradients, Image *color,
                   double ***O, double ***M){
    for(int i = 0; i < color->h; i++){
        for(int j = 0; j < color->w ; j++){
            pii p = {i,j};
            int dy = (int) (gradY[i][j] / gradients[i][j]);
            int dx = (int) (gradX[i][j] / gradients[i][j]);
            for(int r = 1; r < MAX_RADII; r++){
                pii p_plus =  {p.first + dy * r, p.second + dx * r};
                pii p_minus = {p.first - dy * r, p.second - dx * r};

                if(p_plus.first >= 0 && p_plus.first < color->h && p_plus.second >= 0 && p_plus.first < color->w){
                    O[r][p_plus.first][p_plus.second] += 1;
                    M[r][p_plus.first][p_plus.second] += gradients[i][j];
                }
                if(p_minus.first >= 0 && p_minus.first < color->h && p_minus.second >= 0 && p_minus.first < color->w){
                    O[r][p_minus.first][p_minus.second] -= 1;
                    M[r][p_minus.first][p_minus.second] -= gradients[i][j];
                }
            }
        }
    }
}

//gaussian convolve
void gaussConvolve(double ***M, double **postGauss, Image *color){

    double gauss[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    for(int i = 0; i < color->h; i++){
        for(int j = 0; j < color->h; j++){
            double t = 0;
            for(int r = 0; r < MAX_RADII; r++){
                t += M[r][i][j];
            }
            M[0][i][j] = t;
        }
    }
    for(int i = 1; i < color->h - 1; i++){
        for(int j = 1; j < color->h - 1; j++){
            double g = 0;
            for(int dy = 0; dy < 3; dy++){
                for(int dx = 0; dx < 3; dx++){
                    g += M[0][dy+i-1][dx+j-1] * gauss[dy][dx];
                }
            }
            postGauss[i][j] = g;
        }
    }
}

//elipses
// void responseMap(double *****Mg, double *****Og, double **gradX, double **gradY, double **gradientDir){
//     for(int i = 0; i < color->h; i++){
//         for(int j = 0; j < color->w ; j++){
//             pii p = {i,j};
//             int dy = (int) (gradY[i][j] / gradients[i][j]);
//             int dx = (int) (gradX[i][j] / gradients[i][j]);
//             for(int r = 1; r < MAX_RADII; r++){
//                 for(int a = 0; a < n; a++){
//                     for(int b = 0; b < n; b++){

//                     }
//                 }
//                 // pii p_plus =  {p.first + dy * r, p.second + dx * r};
//                 // pii p_minus = {p.first - dy * r, p.second - dx * r};

//                 // if(p_plus.first >= 0 && p_plus.first < color->h && p_plus.second >= 0 && p_plus.first < color->w){
//                 //     O[r][p_plus.first][p_plus.second] += 1;
//                 //     M[r][p_plus.first][p_plus.second] += gradients[i][j];
//                 // }
//                 // if(p_minus.first >= 0 && p_minus.first < color->h && p_minus.second >= 0 && p_minus.first < color->w){
//                 //     O[r][p_minus.first][p_minus.second] -= 1;
//                 //     M[r][p_minus.first][p_minus.second] -= gradients[i][j];
//                 // }
//             }
//         }
//     }
// }

int main(){
    string basefile = "coins";
    string filename = "images/" + basefile + ".txt";
    std::ifstream fin(filename);
    int h, w;
    fin >> h >> w;
    string line;
    Image* color = new Image(h,w);
    // fin >> color->h >> color->w;
    // printf("")
    for(int i = 0; i < color->h; i++){
        for(int j = 0; j < color->w; j++){
            int a, b, c;
            fin >> a >> b >> c;
            color->img[i][j].r = a;
            color->img[i][j].g = b;
            color->img[i][j].b = c;
        }
    }

    printf("%d, %d, %d, \n %d, %d, %d\n", 
            color->img[0][0].r, color->img[0][0].g, color->img[0][0].b,
            color->img[399][0].r, color->img[399][0].g, color->img[399][0].b);

    unsigned long start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    double **grayscaleImage, **gradients, **gradientDir, **nms_grads, **gradX, **gradY;
    grayscaleImage = (double**)calloc(color->h, sizeof(double*));
    for(int i = 0; i < color->h; i++) grayscaleImage[i] = (double*) calloc(color->w, sizeof(double));
    gradients = (double**)calloc(color->h, sizeof(double*));
    for(int i = 0; i < color->h; i++) gradients[i] = (double*) calloc(color->w, sizeof(double));
    gradientDir = (double**)calloc(color->h, sizeof(double*));
    for(int i = 0; i < color->h; i++) gradientDir[i] = (double*) calloc(color->w, sizeof(double));

    nms_grads = (double**)calloc(color->h, sizeof(double*));
    for(int i = 0; i < color->h; i++) nms_grads[i] = (double*) calloc(color->w, sizeof(double));

    gradX = (double**)calloc(color->h, sizeof(double*));
    for(int i = 0; i < color->h; i++) gradX[i] = (double*) calloc(color->w, sizeof(double));
    gradY = (double**)calloc(color->h, sizeof(double*));
    for(int i = 0; i < color->h; i++) gradY[i] = (double*) calloc(color->w, sizeof(double));



    grayscale(color, grayscaleImage);
    ofstream grayfile("images/" + basefile + "_gray.txt");
     for(int i = 0; i < color->h; i++){
        for(int j = 0; j < color->w; j++){
            grayfile << grayscaleImage[i][j] << endl;
        }
    }
    grayfile.close();
    printf("%d, %d, %d, \n %f, %f, %f\n", 
            color->img[0][0].r, color->img[0][0].g, color->img[0][0].b,
           grayscaleImage[399][0], grayscaleImage[399][0], grayscaleImage[399][0]);

    gradient(grayscaleImage, gradients, gradientDir, color, gradX, gradY);

    ofstream gradientfile("images/" + basefile + "_grads.txt");
    printf("writing grads\n");
    for(int i = 0; i < color->h; i++){
        for(int j = 0; j < color->w; j++){
            gradientfile << gradients[i][j] << endl;
        }
    }
    gradientfile.close();

    ofstream grad_dir_file("images/" + basefile + "_gradsdir.txt");
    printf("writing grads\n");
    for(int i = 0; i < color->h; i++){
        for(int j = 0; j < color->w; j++){
            grad_dir_file << gradientDir[i][j] << endl;
        }
    }
    grad_dir_file.close();

    NMS(gradients, gradientDir, nms_grads, color);
    ofstream nms_file("images/" + basefile + "_nmsgrads.txt");
    printf("writing grads\n");
    for(int i = 0; i < color->h; i++){
        for(int j = 0; j < color->w; j++){
            nms_file << nms_grads[i][j] << endl;
        }
    }
    nms_file.close();

    double ***O, ***M;
    O = (double***)calloc(MAX_RADII, sizeof(double**));
    M = (double***)calloc(MAX_RADII, sizeof(double**));
    for(int r = 0; r < MAX_RADII; r++){
        O[r] = (double**)calloc(color->h, sizeof(double*));
        M[r] = (double**)calloc(color->h, sizeof(double*));
        for(int i = 0; i < color->h; i++){
            O[r][i] = (double*)calloc(color->w, sizeof(double));
            M[r][i] = (double*)calloc(color->w, sizeof(double));
        }
    }

    radialSymmetry(gradX, gradY, gradients, color, O, M);
    ofstream rad_sym_file("images/" + basefile + "_radsym.txt");
    printf("writing grads\n");
    for(int i = 0; i < color->h; i++){
        for(int j = 0; j < color->w; j++){
            double tmp_write = 0.0;
            for(int r = 0; r < MAX_RADII; r++){
                tmp_write += M[r][i][j];
            }
            rad_sym_file << tmp_write << endl;
        }
    }
    rad_sym_file.close();

    double **postGauss;
    postGauss = (double**)calloc(color->h, sizeof(double*));
    for(int i = 0; i < color->h; i++) postGauss[i] = (double*) calloc(color->w, sizeof(double));

    gaussConvolve(M, postGauss, color);

    ofstream gauss_file("images/" + basefile + "_gauss.txt");
    printf("writing grads\n");
    for(int i = 0; i < color->h; i++){
        for(int j = 0; j < color->w; j++){
            gauss_file << postGauss[i][j] << endl;
        }
    }
    gauss_file.close();


    unsigned long end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    printf("execution time : %ld\n", end - start);


}
