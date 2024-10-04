#include "arvc_utils/metrics.hpp"

        
arvc::Metrics::Metrics(){};

arvc::Metrics::~Metrics(){};

struct arvc::Metrics::values arvc::Metrics::computeMetricsFromConfusionMatrix(int tp, int fp, int fn, int tn){

    this->values.accuracy = (float)(tp + tn) / (float)(tp + fp + fn + tn);
    this->values.precision = (float)tp / (float)(tp + fp);
    this->values.recall = (float)tp / (float)(tp + fn);
    this->values.f1_score = 2 * (this->values.precision * this->values.recall) / (this->values.precision + this->values.recall);
    this->values.tp = tp; this->values.fp = fp; this->values.fn = fn; this->values.tn = tn;
    return values;
}

template <typename T>
T arvc::Metrics::getMean(std::vector<T> vec){
    T sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    T mean = sum / vec.size();
    return mean;
}


void arvc::Metrics::plotMetrics(){
    std::cout << "Metrics: " << std::endl;
    std::cout << "\tAccuracy: "   << getMean<float>(accuracy) << std::endl;
    std::cout << "\tPrecision: "  << getMean<float>(precision) << std::endl;
    std::cout << "\tRecall: "     << getMean<float>(recall) << std::endl;
    std::cout << "\tF1 Score: "   << getMean<float>(f1_score) << std::endl;
    std::cout << "\tTP: " << getMean<int>(tp) << std::endl;
    std::cout << "\tFP: " << getMean<int>(fp) << std::endl;
    std::cout << "\tFN: " << getMean<int>(fn) << std::endl;
    std::cout << "\tTN: " << getMean<int>(tn) << std::endl;
}

