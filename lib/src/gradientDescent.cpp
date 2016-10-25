#include "gradientDescent.h"

namespace neuralToy
{
    GradientDescent::GradientDescent(const Network& network, Learning learning)
        : LearningRule(network, learning)
    {
        learning_rate_ = 0.5;
        momentum_ = 0.0;

        updated_weight_ = new double**[network_->layers_number()];

        for (unsigned short l = 0; l < network_->layers_number(); ++l) {
            updated_weight_[l] = new double*[network_->get_layer(l)
                                 .neurons_number()];

            for (unsigned short n = 0;
                 n < network_->get_layer(l).neurons_number(); ++n) {
                unsigned short wmax = network_->get_layer(l).get_neuron(n)
                                      .inputs_number() + 1;
                updated_weight_[l][n] = new double[wmax];

                for (unsigned short w = 0; w < wmax; ++w) {
                    updated_weight_[l][n][w] = 0;
                }
            }
        }
    }

    GradientDescent::~GradientDescent()
    {
        for (unsigned short l = 0; l < network_->layers_number(); ++l) {
            for (unsigned short n = 0;
                 n < network_->get_layer(l).neurons_number(); ++n) {
                delete [] updated_weight_[l][n];
                updated_weight_[l][n] = nullptr;
            }

            delete [] updated_weight_[l];
            updated_weight_[l] = nullptr;
        }

        delete [] updated_weight_;
        updated_weight_ = nullptr;
    }

    double GradientDescent::get_learning_rate(void)
    {
        return learning_rate_;
    }

    void GradientDescent::set_learning_rate(double value)
    {
        //assert(0 <= value && 1 >= value);

        learning_rate_ = value;
    }

    double GradientDescent::get_momentum(void)
    {
        return momentum_;
    }

    void GradientDescent::set_momentum(double value)
    {
        assert(0 <= value && 1 >= value);

        momentum_ = value;
    }

    void GradientDescent::update_weight(void)
    {
        for (unsigned short l = 0; l < network_->layers_number(); ++l) {
            for (unsigned short n = 0;
                 n < network_->get_layer(l).neurons_number(); ++n) {
                for (unsigned short w = 0;
                     w < network_->get_layer(l).get_neuron(n)
                     .inputs_number(); ++w) {

                    updated_weight_[l][n][w] =
                        momentum_ * updated_weight_[l][n][w]
                        - learning_rate_ * get_gradient(l, n, w);

                    network_->adjust_layer(l).adjust_neuron(n).set_weight(w,
                            network_->get_layer(l).get_neuron(n)
                            .get_weight(w) + updated_weight_[l][n][w]);
                }
            }
        }
    }

    void GradientDescent::train(const DataSet& ds)
    {
        if (get_learning() == Learning::ONLINE) {
            for (auto it = ds.cbegin(); it != ds.cend(); ++it) {
                calculate_backward_pass(**it);
                update_weight();
            }
        } else if (get_learning() == Learning::BATCH) {
            accumulate_gradients(ds);
            update_weight();
        } else {
            throw;
        }
    }
}
