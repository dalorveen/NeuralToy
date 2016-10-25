#ifndef LEARNINGRULE_H
#define LEARNINGRULE_H

#include "network.h"
#include "dataSet.h"

namespace neuralToy
{
    enum Learning
    {
        ONLINE,
        BATCH,
    };

    class LearningRule
    {
    public:
        LearningRule(const Network&, Learning);
        virtual ~LearningRule();
        Learning get_learning(void) const;
        static double calculate_derivative(const double&, TransferFunction);
        double calculate_mean_squared_error(const DataSet&);
        virtual void train(const DataSet&) = 0;
        Network get_network(void);

    protected:
        Network* network_;
        void calculate_backward_pass(const DataPair&);
        const double& get_error(unsigned short, unsigned short) const;
        const double& get_gradient(unsigned short,
                                   unsigned short,
                                   unsigned short) const;

        void reset_gradients(void);
        void accumulate_gradients(const DataSet&);

    private:
        Learning learning_;

        double** errors_;
        double*** gradients_;

        void calculate_errors_for_output_layer(const DataPair&);
        void calculate_errors_for_hidden_layers(void);
        void calculate_gradients(const DataPair&);
    };
}

#endif // LEARNINGRULE_H
