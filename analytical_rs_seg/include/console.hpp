#ifndef CONSOLE_HPP
#define CONSOLE_HPP

#include <iostream>

namespace utils{
    class Console
    {
        #define RESET   "\033[0m"
        #define RED     "\033[31m"
        #define GREEN   "\033[32m"  
        #define YELLOW  "\033[33m"
        #define BLUE    "\033[34m"

        public:
        Console(){
            enable = true;
            enable_vis = true;
        }

        /**
         * @brief Print an info message to the console.
         * @param msg The message to print.
         * @param color The color of the message. Default is no color.
        */
        inline void info(const std::string msg, const std::string color)
        {
            
            if (color == "RED")
                std::cout << RED;
            else if (color == "GREEN")
                std::cout << GREEN;
            else if (color == "YELLOW")
                std::cout << YELLOW;
            else if (color == "BLUE")
                std::cout << BLUE;
            else 
                std::cout << color;

            std::cout << msg << RESET << std::endl;
        }


        /**
         * @brief Print a warning message to the console.
         * @param msg The message to print.
         * @param color The color of the message. Default is yellow.
        */
        inline void debug(const std::string msg, const std::string color="YELLOW")
        {
            if (this->enable){
                if (color == "RED")
                    std::cout << RED;
                else if (color == "GREEN")
                    std::cout << GREEN;
                else if (color == "YELLOW")
                    std::cout << YELLOW;
                else if (color == "BLUE")
                    std::cout << BLUE;
                else if (color == "WHITE")
                    std::cout << RESET;
                else
                    std::cout << color;

                std::cout << msg << RESET << std::endl;
            }
        }

        bool enable;
        bool enable_vis; 

    };
}

#endif // CONSOLE_HPP