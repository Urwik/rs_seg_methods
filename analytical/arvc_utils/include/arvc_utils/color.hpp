#pragma once

#include <iostream>
#include <pcl/common/common.h>

namespace arvc {

    class Color {
    public:

        float r;
        float g;
        float b;

        /**
         * @brief Default constructor. Initializes the color with random values in the range [0, 255].
         * @param min The minimum value for the color components.
         * @param max The maximum value for the color components.
         * 
        */
        Color(int min = 0, int max = 255);

        /**
         * @brief Constructor with specific color values.
         * @param _r The red component.
         * @param _g The green component.
         * @param _b The blue component.
         * 
        */
        Color(int _r, int _g, int _b);

        /**
         * @brief Set the color to random values in the range [100, 255] by default.
         * @param min The minimum value for the color components.
         * @param max The maximum value for the color components.
         * 
        */
        void random(int min = 100, int max = 255);

        /**
         * @brief Normalize the color values to the range [0,1]
        */
        void normalize();


        /**
         * @brief Convert the color to a pcl::RGB object.
        */
        pcl::RGB toPclRGB();


    };


   inline std::ostream& operator<<(std::ostream& os, const Color& c) {
        os << "[ " << c.r << ", " << c.g << ", " << c.b << " ]" << std::endl;
        return os;
    }

} // namespace arvc

    // std::ostream& operator<<(std::ostream& os, const arvc::Color& c);

const arvc::Color RED_COLOR = arvc::Color(255,0,0);
const arvc::Color BLUE_COLOR = arvc::Color(0,0,255);
const arvc::Color GREEN_COLOR = arvc::Color(0,255,0);
const arvc::Color YELLOW_COLOR = arvc::Color(255,255,0);
const arvc::Color WHITE_COLOR = arvc::Color(255,255,255);
const arvc::Color BLACK_COLOR = arvc::Color(0,0,0);
const arvc::Color ORANGE_COLOR = arvc::Color(255,165,0);
const arvc::Color PURPLE_COLOR = arvc::Color(128,0,128);
const arvc::Color PINK_COLOR = arvc::Color(255,192,203);
const arvc::Color GREY_COLOR = arvc::Color(128,128,128);