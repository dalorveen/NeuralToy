#ifndef RESILIENTPROPAGATION_H
#define RESILIENTPROPAGATION_H

#include <algorithm>

#include "learningRule.h"
#include "network.h"
#include "dataSet.h"

namespace neuralToy
{
    class ResilientPropagation : public LearningRule
    {
    public:
        ResilientPropagation(const Network&, double = 0.0125);
        virtual ~ResilientPropagation();
        int sign(double);
        void train(const DataSet&);

        double get_initial_delta(void) const;
        void set_initial_delta(double);

        double get_eta_minus(void) const;
        void set_eta_minus(double);

        double get_eta_plus(void) const;
        void set_eta_plus(double);

        double get_delta_min(void) const;
        void set_delta_min(double);

        double get_delta_max(void) const;
        void set_delta_max(double);

    protected:

    private:
        double initial_delta_;
        double eta_minus_;
        double eta_plus_;
        double delta_min_;
        double delta_max_;

        struct Branch
        {
            double gradient_previous;
            double delta;
            double dw;
        } ***branch_;

        double mse_;
        double mse_previous_;
    };
}

#endif // RESILIENTPROPAGATION_H
