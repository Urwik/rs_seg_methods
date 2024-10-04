#pragma once

#include <iostream>
#include <chrono>

namespace arvc{
  class Timer
  {
  private:
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point stop;
    double duration;

  public:
    // Default constructor
    Timer(/* args */);


    // Start the timer
    void start_timer();


    // Stop the timer
    void stop_timer();


    /**
     * @brief Get the duration of the timer.
     * @return The duration of the timer.
    */
    int get_duration();


    /**
     * @brief Print the duration of the timer.
    */
    void print_duration(const std::string& _message = "");


    /**
     * @brief Print the current duration of the timer.
    */
    void print_current_duration(const std::string& _message = "");

  };
} // namespace arvc