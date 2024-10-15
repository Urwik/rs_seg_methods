#ifndef METRICS_HPP
#define METRICS_HPP

#include <iostream>

namespace utils {
    class Metrics
    {
        public:
        int tp;
        int tn;
        int fp;
        int fn;

        Metrics() {
            this->tp = 0;
            this->tn = 0;
            this->fp = 0;
            this->fn = 0;
        };

        inline float precision() {
            return (float)this->tp / (float)(this->tp + this->fp);
        }

        inline float recall() {
            return (float)this->tp / (float)(this->tp + this->fn);
        }

        inline float f1_score() {
            return 2 * (this->precision() * this->recall()) / (this->precision() + this->recall());
        }

        inline float accuracy() {
            return (float)(this->tp + this->tn) / (float)(this->tp + this->fp + this->fn + this->tn);
        }

        inline float miou() {
            return (float)this->tp / (float)(this->tp + this->fp + this->fn);
        }

        inline void plotMetrics(){
            // std::cout << "Metrics: " << std::endl;
            std::cout << "\tPrecision: "  << this->precision() << std::endl;
            std::cout << "\tRecall: "     << this->recall() << std::endl;
            std::cout << "\tF1 Score: "   << this->f1_score() << std::endl;
            std::cout << "\tAccuracy: "   << this->accuracy() << std::endl;
            std::cout << "\tMIoU: "       << this->miou() << std::endl;
        }
    };
}

#endif // METRICS_HPP