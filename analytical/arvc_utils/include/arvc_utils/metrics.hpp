#pragma once

#include <iostream>
#include <vector>
#include <numeric>

namespace arvc{
    class Metrics{

    public:
        struct values{
            float accuracy;
            float precision;
            float recall;
            float f1_score;
            int tp;
            int fp;
            int fn;
            int tn;
        };


        // Default constructor
        Metrics();
        ~Metrics();


        /**
         * @brief Compute the metrics from the confusion matrix.
         * @param tp True Positives
         * @param fp False Positives
         * @param fn False Negatives
         * @param tn True Negatives
         * @return The metrics values.
        */
        values computeMetricsFromConfusionMatrix(int tp, int fp, int fn, int tn);


        /**
         * @brief Compute the mean of a vector.
         * @param vec The vector to compute the mean.
         * @return The mean of the vector.
         * @usage float mean = getMean<float>(vec);
        */
        template <typename T>
        T getMean(std::vector<T> vec);


        // Plot the metrics
        void plotMetrics();


        std::vector<float> accuracy;
        std::vector<float> precision;
        std::vector<float> recall;
        std::vector<float> f1_score;
        std::vector<int> tp;
        std::vector<int> fp;
        std::vector<int> fn;
        std::vector<int> tn;
        values values;  
    };
} // namespace arvc