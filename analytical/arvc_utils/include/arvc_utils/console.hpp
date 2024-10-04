#pragma once

#include <iostream>

#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"  
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"

namespace arvc{

    class Console
    {
    public:
        Console();

        /**
         * @brief Print an info message to the console.
         * @param msg The message to print.
         * @param color The color of the message. Default is no color.
        */
        void info(const std::string msg, const std::string color = "");

        /**
         * @brief Print a warning message to the console.
         * @param msg The message to print.
         * @param color The color of the message. Default is yellow.
        */
        void debug(const std::string msg, const std::string color = "YELLOW");

        bool enable;
        bool enable_vis; 
    };
} // namespace arvc