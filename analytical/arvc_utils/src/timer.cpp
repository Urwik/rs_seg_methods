#include "arvc_utils/timer.hpp"


arvc::Timer::Timer(){
    this->duration = 0.0;
}



void arvc::Timer::start_timer() {
    this->start = std::chrono::high_resolution_clock::now();
}


void arvc::Timer::stop_timer() {
    this->stop = std::chrono::high_resolution_clock::now();
}


int arvc::Timer::get_duration() {
    this->duration = std::chrono::duration_cast<std::chrono::milliseconds>(this->stop - this->start).count();
    return this->duration;
}


void arvc::Timer::print_duration(const std::string& _message) {
    this->get_duration();
    std::cout << _message <<" exec Time: " << this->duration << " ms" << std::endl;
}

void arvc::Timer::print_current_duration(const std::string& _message) {
    std::chrono::high_resolution_clock::time_point current = std::chrono::high_resolution_clock::now();
    double current_duration = std::chrono::duration_cast<std::chrono::milliseconds>(current - this->start).count();

    std::cout << _message <<" exec Time: " << current_duration << " ms" << std::endl;
}