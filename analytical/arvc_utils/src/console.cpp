#include "arvc_utils/console.hpp"



arvc::Console::Console()
{
    enable = true;
    enable_vis = true;
}



void arvc::Console::info(const std::string msg, const std::string color)
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



void arvc::Console::debug(const std::string msg, const std::string color)
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
