#include <bits/stdc++.h>
using namespace std;

//pi
#define PI 3.14159265358979323846
//max a/b or radii for fst
#define MAX_RADII 50
//scaling factor from paper
#define Kn 9.9
#define alpha 1.0
//M response map threshold
#define THRESH 1000
//not iterating through a lot of angles lmao
//in degrees
#define angular_granularity 60

#define LOG true

typedef pair<int,int> pii;

const vector<int> a_vals = {2, 4, 6, 8};
const vector<int> b_vals = {2, 4, 6, 8};

typedef struct Pixel {
    uint8_t r,g,b;
} Pixel;

const int M[2][2] = {
    {0, 1}, {-1, 0}
};

const int M_inv[2][2] = {
    {0, -1}, {1, 0}
};


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

typedef struct Params {
    int theta, a, b;
} Params;


void matInv(double (&A)[2][2], double (&A_inv)[2][2]){
    float det = 1 / (A[0][0] * A[1][1] - A[0][1] * A[1][0]);
    A_inv[0][0] = A[1][1] * det;
    A_inv[0][1] = -A[0][1] * det;
    A_inv[1][0] = -A[1][0] * det;
    A_inv[1][1] = A[0][0] * det;
}

void M_mults(double (&M)[2][2], double (&res)[2][2]){
    double M_inv[2][2];
    matInv(M, M_inv);
    res[0][0] = M[0][0] * M_inv[1][1] - M[0][1] * M_inv[0][1];
    res[0][1] = M[0][1] * M_inv[0][0] - M[0][0] * M_inv[1][0] ;
    res[1][0] = M[1][0] * M_inv[1][1] - M[1][1] * M_inv[0][1];
    res[1][1] = M[1][1] * M_inv[0][0] - M[1][0] * M_inv[1][0];
}

int clamp(int min, int v, int high){
    if(v < min)return min;
    if(v > high) return high;
    return v;
}

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
            if(gradients[i][j] == 0){
                continue;
            }
            // printf("i,j: %d, %d\n", i, j);
            pii p = {i,j};
            double dy =  (gradY[i][j] / gradients[i][j]);
            double dx =  (gradX[i][j] / gradients[i][j]);
            // printf("dy, dx: %f, %f, %f, %f\n", dy, dx, gradY[i][j], gradX[i][j] );
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
void gaussConvolve(double ***M, double **postGauss, int **radii, Image *color){

    double gauss[3][3] = {
        {1.0/16.0, 2.0/16.0, 1.0/16.0},
        {2.0/16.0, 4.0/16.0, 2.0/16.0},
        {1.0/16.0, 2.0/16.0, 1.0/16.0}
    };

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

void ellipseResponseMap(double *****Mg, double *****Og, double **gradX, double **gradY, double **gradients, double **gradientDir, Image *color){
    for(int i = 0; i < color->h; i++){
        for(int j = 0; j < color->w; j++){
            if(gradients[i][j] == 0){
                continue;
            }
            pii p = {i,j};
            double dy = (gradY[i][j] / gradients[i][j]);
            double dx = (gradX[i][j] / gradients[i][j]);
            // printf("(%d, %d) -> {%f, %f} |%f|\n", i, j, gradY[i][j], gradX[i][j], gradients[i][j]);
            // printf("processing pixel %d, %d\n", i, j);
            // for(int degr = 0; degr < 360; degr += angular_granularity){
            for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++) {
                int degr = degr_idx * angular_granularity;
                double theta = degr * PI/180.0;
                for(int a_idx = 1; a_idx < a_vals.size(); a_idx++){
                    for(int b_idx = 1; b_idx < b_vals.size(); b_idx++){
                        int a = a_vals[a_idx];
                        int b = b_vals[b_idx];
                        double G[2][2] = {
                            {a * cos(theta), -b * sin(theta)},
                            {a * sin(theta),  b * cos(theta)},
                        };
                        // printf("generated G matrix = [ [%f, %f], [%f, %f]]\n", G[0][0], G[0][1], G[1][0], G[1][1]);
                        double transform_matrix[2][2];
                        M_mults(G, transform_matrix);
                        // printf("(%f,%d,%d)generated T matrix = [ [%f, %f], [%f, %f]]\n", 
                                // theta, a, b, transform_matrix[0][0], transform_matrix[0][1], transform_matrix[1][0], transform_matrix[1][1]);

                        pair<double, double> grad_t = {
                            dy * transform_matrix[0][0] + dx * transform_matrix[0][1],
                            dy * transform_matrix[1][0] + dx * transform_matrix[1][1]
                        };

                        double grad_mag = sqrt(grad_t.first * grad_t.first + grad_t.second * grad_t.second);
                        grad_t.first /= grad_mag;
                        grad_t.second /= grad_mag;

                        // printf("generated transformed gradients {%f, %f} from {%f, %f}\n", grad_t.first, grad_t.second, dy, dx);
                        for(int n = 1; n < MAX_RADII; n++){
                            pii p_plus =  {p.first + grad_t.first * n, p.second + grad_t.second * n};
                            pii p_minus = {p.first - grad_t.first * n, p.second - grad_t.second * n};
                            // printf("for (theta,a,b) = (%d,%d,%d), have points +(%d,%d) -(%d,%d)\n",
                            //         degr, a, b, p_plus.first, p_plus.second, p_minus.first, p_minus.second);
                            if(p_plus.first >= 0 && p_plus.first < color->h && p_plus.second >= 0 && p_plus.first < color->w){
                                Og[degr_idx][a_idx][b_idx][p_plus.first][p_plus.second] += 1;
                                Mg[degr_idx][a_idx][b_idx][p_plus.first][p_plus.second] += gradients[i][j];
                            }
                            if(p_minus.first >= 0 && p_minus.first < color->h && p_minus.second >= 0 && p_minus.first < color->w){
                                Og[degr_idx][a_idx][b_idx][p_minus.first][p_minus.second] -= 1;
                                Mg[degr_idx][a_idx][b_idx][p_minus.first][p_minus.second] -= gradients[i][j];
                            }
                        }
                        
                    }
                }
            }
        }
    }
}

void generate_S(double *****M, double *****O, double *****S, double *****S_nms,
                double **S_flat, double **S_flat_nms, Image* color){

    double gauss[3][3] = {
        {1.0/16.0, 2.0/16.0, 1.0/16.0},
        {2.0/16.0, 4.0/16.0, 2.0/16.0},
        {1.0/16.0, 2.0/16.0, 1.0/16.0}
    };

    for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++){
        int degr = degr_idx * angular_granularity;
        for(int a_idx = 0; a_idx < a_vals.size(); a_idx++){
            int a = a_vals[a_idx];
            for(int b_idx = 0; b_idx < b_vals.size(); b_idx++){
                int b = b_vals[b_idx];
                for(int i = 0; i < color->h; i++){
                    for(int j = 0; j < color->w; j++){
                        double o_hat = max(O[degr_idx][a_idx][b_idx][i][j], Kn);

                        O[degr_idx][a_idx][b_idx][i][j] = (abs(o_hat) / Kn) 
                                                          * M[degr_idx][a_idx][b_idx][i][j] / Kn;
                    }
                }
            }
        }
    }

    for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++){
        for(int a_idx = 0; a_idx < a_vals.size(); a_idx++){
            for(int b_idx = 0; b_idx < b_vals.size(); b_idx++){
                for(int i = 1; i < color->h - 1; i++){
                    for(int j = 1; j < color->w - 1; j++){
                        for(int dy = 0; dy < 3; dy++){
                            for(int dx = 0; dx < 3; dx++){
                                S[degr_idx][a_idx][b_idx][i][j] += M[degr_idx][a_idx][b_idx][dy+i-1][dx+j-1] * gauss[dy][dx];
                            }
                        }
                    }
                }
            }
        }
    }

    for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++){
        for(int a_idx = 0; a_idx < a_vals.size(); a_idx++){
            for(int b_idx = 0; b_idx < b_vals.size(); b_idx++){
                for(int i = 1; i < color->h - 1; i++){
                    for(int j = 1; j < color->w - 1; j++){
                        double maxNeighbor = 0.0;
                        for(int dy = 0; dy < 3; dy++){
                            for(int dx = 0; dx < 3; dx++){
                                if(dy != 1 || dx != 1) maxNeighbor = max(maxNeighbor, S[degr_idx][a_idx][b_idx][i+dy-1][j+dx-1]);
                            }
                        }
                        if(S[degr_idx][a_idx][b_idx][i][j] < maxNeighbor) S_nms[degr_idx][a_idx][b_idx][i][j] = 0;
                        else S_nms[degr_idx][a_idx][b_idx][i][j] = S[degr_idx][a_idx][b_idx][i][j];
                    }
                }
            }
        }
    }

    //flatten 5d->2d
    for(int i = 0; i < color->h; i++){
        for(int j = 0; j < color->w; j++){
            double maxresponse = 0.0;
            for(int degr_idx = 0; degr_idx < 360/angular_granularity; degr_idx++){
                for(int a_idx = 0; a_idx < a_vals.size(); a_idx++){
                    for(int b_idx = 0; b_idx < b_vals.size(); b_idx++){
                        maxresponse = max(maxresponse, S[degr_idx][a_idx][b_idx][i][j]);
                    }
                }
            }
            S_flat_nms[i][j] = maxresponse;
        }
    }
    //gassuain blur flat -> blur matrix
    for(int i = 1; i < color->h-1; i++){
        for(int j = 1; j < color->w-1; j++){
            for(int dy = 0; dy < 3; dy++){
                for(int dx = 0; dx < 3; dx++){
                    S_flat[i][j] += S_flat_nms[dy+i-1][dx+j-1] * gauss[dy][dx];
                }
            }
        }
    }
    //nms blur -> flat
    for(int i = 1; i < color->h-1; i++){
        for(int j = 1; j < color->w-1; j++){
            double maxNeighbor = 0.0;
            for(int dy = 0; dy < 3; dy++){
                for(int dx = 0; dx < 3; dx++){
                    if(dy != 1 || dx != 1) maxNeighbor = max(maxNeighbor, S_flat[i+dy-1][j + dx -1]);
                }
            }
            if(maxNeighbor > S_flat[i][j]) S_flat_nms[i][j] = 0;
            else S_flat_nms[i][j] = S_flat[i][j];
        }
    }
}

typedef struct {
    float x;
    float y;
    float r;
} Circle;

typedef struct {
    int A;
    int B;
} Tuple;


Circle* getCircles(double*** M) {
    // count number of circles
    int count = 0;
    int h = sizeof(M[0]) / sizeof(M[0][0]);
    int w = sizeof(M[0][0]) / sizeof(M[0][0][0]);
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            double tmp = 0.0;
            for(int r = 0; r < MAX_RADII; r++){
                tmp = max(M[r][i][j], tmp);
                
                // if(tmp_write) printf("TMP: %f\n", tmp_write);
            }
            if (tmp > 0.0) {
                count ++;
            }
        }
    }
    // use count to generate array
    Circle* circles = new Circle[count];
    int curr = 0;
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            double tmp = 0.0;
            int bestR = 0;
            for(int r = 0; r < MAX_RADII; r++){
                if( M[r][i][j] > tmp){
                    tmp = M[r][i][j];
                    bestR = r;
                }
            }
            if (tmp) {
                circles[curr].y = i;
                circles[curr].x = j;
                circles[curr].r = bestR;
                curr++;
            }
            
        }
    }
    return circles;
}

bool circleCollision(Circle A, Circle B) {
    double distance = sqrt((B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y));
    return distance < (A.r + B.r);
}

Tuple* getCollidingCircles(Circle* circles) {
    int n = sizeof(circles) / sizeof(circles[0]);
    int count = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (circleCollision(circles[i], circles[j])) {
                count++;
            }
        }
    }

    Tuple* tuples = new Tuple[count];
    int curr = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (circleCollision(circles[i], circles[j])) {
                tuples[curr].A = i;
                tuples[curr].B = j;
                curr++;
            }
        }
    }
    return tuples;
}


// Circle checkOnTop

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
    printf("file read, (%d x %d), %d\n", color->h, color->w,(h-1));
    // printf("%d, %d, %d, \n %d, %d, %d\n", 
    //         color->img[0][0].r, color->img[0][0].g, color->img[0][0].b,0, 0, 0);
    //         // color->img[h-1][0].r, color->img[h-1][0].g, color->img[h-1][0].b);

    unsigned long start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    double **grayscaleImage, **gradients, **gradientDir, **nms_grads, **gradX, **gradY, **gaussNms;
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

    int **best_radii;
    best_radii = (int**)calloc(color->h, sizeof(int*));
    for(int i = 0; i < color->h; i++) best_radii[i] = (int*) calloc(color->w, sizeof(int));

    printf("arrs init\n");

    grayscale(color, grayscaleImage);
    printf("grayscaled\n");
    if(LOG){
         ofstream grayfile("images/" + basefile + "_gray.txt");
        for(int i = 0; i < color->h; i++){
            for(int j = 0; j < color->w; j++){
                grayfile << grayscaleImage[i][j] << endl;
            }
        }
        grayfile.close();
    }
   
    // printf("%d, %d, %d, \n %f, %f, %f\n", 
    //         color->img[0][0].r, color->img[0][0].g, color->img[0][0].b,
    //        grayscaleImage[h-1][0], grayscaleImage[h-1][0], grayscaleImage[h-1][0]);
    printf("starting gradient calc\n");
    gradient(grayscaleImage, gradients, gradientDir, color, gradX, gradY);
    printf("finished calculating gradients\n");
    if(LOG){
        ofstream gradientfile("images/" + basefile + "_grads.txt");
        printf("writing grads\n");
        for(int i = 0; i < color->h; i++){
            for(int j = 0; j < color->w; j++){
                gradientfile << gradients[i][j] << endl;
            }
        }
        gradientfile.close();

        ofstream xgradfile("images/" + basefile + "_xgr.txt");
        ofstream ygradfile("images/" + basefile + "_ygr.txt");
        printf("writing grads\n");
        for(int i = 0; i < color->h; i++){
            for(int j = 0; j < color->w; j++){
                xgradfile << gradX[i][j] << endl;
                ygradfile << gradY[i][j] << endl;
            }
        }
        xgradfile.close();
        ygradfile.close();


        ofstream grad_dir_file("images/" + basefile + "_gradsdir.txt");
        printf("writing grads\n");
        for(int i = 0; i < color->h; i++){
            for(int j = 0; j < color->w; j++){
                grad_dir_file << gradientDir[i][j] << endl;
            }
        }
        grad_dir_file.close();
    }
        
    printf("starting calculating NMS\n");
    NMS(gradients, gradientDir, nms_grads, color);
    printf("finished calculating NMS\n");
    if(LOG){
        ofstream nms_file("images/" + basefile + "_nmsgrads.txt");
        printf("writing grads\n");
        for(int i = 0; i < color->h; i++){
            for(int j = 0; j < color->w; j++){
                nms_file << nms_grads[i][j] << endl;
            }
        }
        nms_file.close();
    }
    
    printf("allocating O/M\n");
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
    printf("finished allocating O/M\n");

    printf("allocating Og/Mg\n");

    bool print_alloc = false;

    double *****Mg = (double*****) calloc(360/angular_granularity, sizeof(double****));
    double *****Og = (double*****) calloc(360/angular_granularity, sizeof(double****));
    double *****S  = (double*****) calloc(360/angular_granularity, sizeof(double****));
    double *****S_nms  = (double*****) calloc(360/angular_granularity, sizeof(double****));
    for(int i = 0; i < 360/angular_granularity; i++){
        Mg[i] = (double****) calloc(a_vals.size(), sizeof(double***));
        Og[i] = (double****) calloc(a_vals.size(), sizeof(double***));
        S[i]  = (double****) calloc(a_vals.size(), sizeof(double***));
        S_nms[i] = (double****) calloc(a_vals.size(), sizeof(double***));
        for(int a = 0; a < a_vals.size(); a++){
            Mg[i][a] = (double***) calloc(b_vals.size(), sizeof(double**));
            Og[i][a] = (double***) calloc(b_vals.size(), sizeof(double**));
            S[i][a] = (double***) calloc(b_vals.size(), sizeof(double**));
            S_nms[i][a] = (double***) calloc(b_vals.size(), sizeof(double**));
            for(int b = 0; b < b_vals.size(); b++){
                Mg[i][a][b] = (double**) calloc(color->h, sizeof(double*));
                Og[i][a][b] = (double**) calloc(color->h, sizeof(double*));
                S[i][a][b] = (double**) calloc(color->h, sizeof(double*));
                S_nms[i][a][b] = (double**) calloc(color->h, sizeof(double*));
                for(int y = 0; y < color->h; y++){
                    Mg[i][a][b][y] = (double*) calloc(color->w, sizeof(double));
                    Og[i][a][b][y] = (double*) calloc(color->w, sizeof(double));
                    S[i][a][b][y] = (double*) calloc(color->w, sizeof(double));
                    S_nms[i][a][b][y] = (double*) calloc(color->w, sizeof(double));
                    if(print_alloc)printf("allocated (angle,a, b, y) value (%d, %d, %d, %d)\n", i, a, b, y);
                }
            }
        }
    }

    double **S_flat, **S_flat_nms;
    S_flat = (double**)calloc(color->h, sizeof(double*));
    S_flat_nms = (double**)calloc(color->h, sizeof(double*));

    for(int i = 0; i < color->h; i++){
        S_flat[i] = (double*)calloc(color->w, sizeof(double));
        S_flat_nms[i] = (double*)calloc(color->w, sizeof(double));
    }


    printf("finished allocating Og/Mg\n");
    printf("starting ellipse\n");
    ellipseResponseMap(Mg, Og, gradX, gradY, nms_grads, gradientDir, color);
    printf("finished ellipse responses\n");
    if(LOG){
        ofstream gen_rad_sym_file("images/" + basefile + "_genradsym.txt");
        printf("writing gfrst results");
        for(int i = 0; i < color->h; i++){
            for(int j = 0; j < color->w; j++){
                double tmp_write = 0.0;
                Params opt_param;
                opt_param.a = 0;
                opt_param.b = 0;
                opt_param.theta = 0;
                for(int angle = 0; angle < 360/angular_granularity; angle++){
                    int real_angle = angle * angular_granularity;
                    for(int a = 0; a < a_vals.size(); a++){
                        for(int b = 0; b < b_vals.size(); b++){
                            // printf("(%d, %d) - (%d, %d, %d)\n", i ,j, angle, a, b);
                            if(Mg[angle][a][b][i][j] > tmp_write){
                                opt_param.a = a_vals[a]; opt_param.b = b_vals[b]; opt_param.theta = real_angle;
                                tmp_write = Mg[angle][a][b][i][j];
                            }
                            // tmp_write = max(Mg[angle][a][b][i][j], tmp_write);
                        }
                    }
                }
                gen_rad_sym_file << tmp_write << "," 
                                 << opt_param.a << ","
                                 << opt_param.b << ","
                                 << opt_param.theta << endl;
            }
        }
        gen_rad_sym_file.close();
    }

    printf("starting ellipse gauss + nms\n");
    generate_S(Mg, Og, S, S_nms, S_flat, S_flat_nms, color);
    printf("finished ellipse gauss + nms\n");
    if(LOG){
        printf("writing post-proc\n");
        ofstream radsym_scaled("images/" + basefile + "_gen_scaled.txt");
        ofstream radsym_scaled_nms("images/" + basefile + "_gen_nms.txt");
        for(int i = 0; i < color->h; i++){
            for(int j = 0; j < color->w; j++){
                double scaled_write = 0.0;
                double nms_write = 0.0;
                for(int angle = 0; angle < 360/angular_granularity; angle++){
                    for(int a = 0; a < a_vals.size(); a++){
                        for(int b = 0; b < b_vals.size(); b++){
                            scaled_write = max(scaled_write, S[angle][a][b][i][j]);
                            nms_write    = max(nms_write, S_nms[angle][a][b][i][j]);
                        }
                    }
                }
                radsym_scaled << scaled_write << endl;
                radsym_scaled_nms << nms_write << endl;
            }
        }
        radsym_scaled.close();
        radsym_scaled_nms.close();

        ofstream radsym_scaled_flat("images/" + basefile + "_gen_scaled_flat.txt");
        ofstream radsym_scaled_flat_nms("images/" + basefile + "_gen_scaled_flatnms.txt");
         for(int i = 0; i < color->h; i++){
            for(int j = 0; j < color->w; j++){
                radsym_scaled_flat << S_flat[i][j] << endl;
                radsym_scaled_flat_nms << S_flat_nms[i][j] << endl;
            }
         }
         radsym_scaled_flat.close(); radsym_scaled_flat_nms.close();
    }


    printf("starting circular symm\n");
    radialSymmetry(gradX, gradY, nms_grads, color, O, M);
    printf("finished circular symm\n");
    if(LOG){
        ofstream rad_sym_file("images/" + basefile + "_radsym.txt");
        printf("writing grads\n");
        for(int i = 0; i < color->h; i++){
            for(int j = 0; j < color->w; j++){
                double tmp_write = 0.0;
                for(int r = 0; r < MAX_RADII; r++){
                    tmp_write = max(M[r][i][j], tmp_write);
                    // if(tmp_write) printf("TMP: %f\n", tmp_write);
                }
                rad_sym_file << tmp_write << endl;
            }
        }
        rad_sym_file.close();
    }
   

    double **postGauss;
    postGauss = (double**)calloc(color->h, sizeof(double*));
    for(int i = 0; i < color->h; i++) postGauss[i] = (double*) calloc(color->w, sizeof(double));
    printf("starting guassian blur\n");
    gaussConvolve(M, postGauss, best_radii, color);
    printf("finished guassian blur\n");
    if(LOG){
        ofstream gauss_file("images/" + basefile + "_gauss.txt");
        printf("writing grads\n");
        for(int i = 0; i < color->h; i++){
            for(int j = 0; j < color->w; j++){
                gauss_file << postGauss[i][j] << endl;
            }
        }
        gauss_file.close();
    }
    gaussNms = (double**)calloc(color->h, sizeof(double*));
    for(int i = 0; i < color->h; i++) gaussNms[i] = (double*) calloc(color->w, sizeof(double));

    
    printf("starting guassin blur nms\n");
    postGaussNMS(postGauss, gaussNms, best_radii, color);
    printf("finished guassin blur nms\n");
    if(LOG){
        ofstream post_gauss_nms("images/" + basefile + "_gaussnms.txt");
        for(int i = 0; i < color->h; i++){
            for(int j = 0; j < color->w; j++){
                post_gauss_nms << gaussNms[i][j] << endl;
            }
        }
        post_gauss_nms.close();

        ofstream radii_file("images/" + basefile + "_radii.txt");
        for(int i = 0; i < color->h; i++){
            for(int j = 0; j < color->w; j++){
                radii_file << best_radii[i][j] << endl;
            }
        }
        radii_file.close();
    }
    
    unsigned long end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    printf("execution time : %ld\n", end - start);


}
