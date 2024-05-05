#include <bits/stdc++.h>
using namespace std;
//pi
#define PI 3.14159265358979323846
//max a/b or radii for fst
#define MAX_RADII 25
//scaling factor from paper
#define Kn 9.9
//M response map threshold
#define THRESH 1000


typedef std::pair<int,int> pii;

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



int main(){
    string basefile = "coin_1";
    string filename = "images/" + basefile + ".txt";
    std::ifstream fin(filename);
    int h, w;
    fin >> h >> w;
    string line;
    Image* color = new Image(h,w);
    for(int i = 0; i < color->h; i++){
        for(int j = 0; j < color->w; j++){
            int a, b, c;
            fin >> a >> b >> c;
            color->img[i][j].r = a;
            color->img[i][j].g = b;
            color->img[i][j].b = c;
        }
    }

    

}



