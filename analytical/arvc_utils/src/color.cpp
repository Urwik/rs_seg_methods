#include "arvc_utils/color.hpp"


arvc::Color::Color(int min, int max) {
    
    this->random(min, max);
}

arvc::Color::Color(int _r, int _g, int _b) {

    this->r = _r;
    this->g = _g;
    this->b = _b;
}

void arvc::Color::random(int min, int max) {

    this->r = min + rand() % (max - min + 1);
    this->g = min + rand() % (max - min + 1);
    this->b = min + rand() % (max - min + 1);
}


void arvc::Color::normalize() {

    this->r = (float) this->r / 255;
    this->g = (float) this->g / 255;
    this->b = (float) this->b / 255;
}


pcl::RGB arvc::Color::toPclRGB(){
    
    pcl::RGB color;
    int r = static_cast<int>(this->r);
    int g = static_cast<int>(this->g);
    int b = static_cast<int>(this->b);

    color.r = r;
    color.g = g;
    color.b = b;

    return color;
}