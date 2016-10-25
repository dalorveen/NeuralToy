#include "resilientpropagation.h"

namespace neuralToy
{
    ResilientPropagation::ResilientPropagation(const Network& network,
                                               double initial_delta)
        : LearningRule(network, Learning::BATCH)
    {
        initial_delta_ = initial_delta;
        eta_minus_ = 0.5;
        eta_plus_ = 1.2;
        delta_min_ = 0.000001;
        delta_max_ = 50.0;

        branch_ = new Branch**[network.layers_number()];

        for (unsigned short l = 0; l < network.layers_number(); ++l) {
            branch_[l] = new Branch*[network.get_layer(l).neurons_number()];

            for (unsigned short n = 0;
                 n < network.get_layer(l).neurons_number(); ++n) {
                unsigned short wmax = network.get_layer(l).get_neuron(n)
                                      .inputs_number();
                branch_[l][n] = new Branch[wmax];

                for (unsigned short w = 0; w < wmax; ++w) {
                    branch_[l][n][w].gradient_previous = 0;
                    branch_[l][n][w].delta = initial_delta_;
                    branch_[l][n][w].dw = 0;
                }
            }
        }

        mse_ = 0;
        mse_previous_ = 0;
    }

    ResilientPropagation::~ResilientPropagation()
    {
        for (unsigned short l = 0; l < network_->layers_number(); ++l) {
            for (unsigned short n = 0;
                 n < network_->get_layer(l).neurons_number(); ++n) {
                delete [] branch_[l][n];
            }

            delete [] branch_[l];
            branch_[l] = nullptr;
        }

        delete [] branch_;
        branch_ = nullptr;
    }

    double ResilientPropagation::get_initial_delta(void) const
    {
        return initial_delta_;
    }

    void ResilientPropagation::set_initial_delta(double value)
    {
        initial_delta_ = value;
    }

    double ResilientPropagation::get_eta_minus(void) const
    {
        return eta_minus_;
    }

    void ResilientPropagation::set_eta_minus(double value)
    {
        assert(0 < value && value < 1);

        eta_minus_ = value;
    }

    double ResilientPropagation::get_eta_plus(void) const
    {
        return eta_plus_;
    }

    void ResilientPropagation::set_eta_plus(double value)
    {
        assert(1 < value);

        eta_plus_ = value;
    }

    double ResilientPropagation::get_delta_min(void) const
    {
        return delta_min_;
    }

    void ResilientPropagation::set_delta_min(double value)
    {
        delta_min_ = value;
    }

    double ResilientPropagation::get_delta_max(void) const
    {
        return delta_max_;
    }

    void ResilientPropagation::set_delta_max(double value)
    {
        delta_max_ = value;
    }

    int ResilientPropagation::sign(double value)
    {
        return (0 < value) - (value < 0);
    }

    void ResilientPropagation::train(const DataSet& ds)
    {
        accumulate_gradients(ds);
        mse_ = calculate_mean_squared_error(ds);

        for (unsigned short l = 0; l < network_->layers_number(); ++l) {
            for (unsigned short n = 0;
                 n < network_->get_layer(l).neurons_number(); ++n) {
                for (unsigned short w = 0;
                     w < network_->get_layer(l).get_neuron(n)
                     .inputs_number(); ++w) {
                    double pd = branch_[l][n][w].gradient_previous
                                * get_gradient(l, n, w);
                    if (pd > 0) {
                        branch_[l][n][w].delta
                            = std::min(branch_[l][n][w].delta * eta_plus_,
                                       delta_max_);

                        branch_[l][n][w].dw
                            = -sign( get_gradient(l, n, w) )
                              * branch_[l][n][w].delta;

                        network_->adjust_layer(l).adjust_neuron(n)
                            .set_weight(w, network_->get_layer(l).get_neuron(n)
                                        .get_weight(w) + branch_[l][n][w].dw);

                        branch_[l][n][w].gradient_previous
                            = get_gradient(l, n, w);
                    } else if (pd < 0) {
                        branch_[l][n][w].delta
                            = std::max(branch_[l][n][w].delta
                                       * eta_minus_, delta_min_);

                        if (mse_ > mse_previous_) {
                            network_->adjust_layer(l).adjust_neuron(n)
                                .set_weight(w, network_->get_layer(l)
                                            .get_neuron(n).get_weight(w)
                                            - branch_[l][n][w].dw);
                        }

                        branch_[l][n][w].gradient_previous = 0;
                    } else {
                        branch_[l][n][w].dw = -sign( get_gradient(l, n, w) )
                                              * branch_[l][n][w].delta;

                        network_->adjust_layer(l).adjust_neuron(n)
                            .set_weight(w, network_->get_layer(l)
                                        .get_neuron(n).get_weight(w)
                                        + branch_[l][n][w].dw);

                        branch_[l][n][w].gradient_previous
                            = get_gradient(l, n, w);
                    }
                }
            }
        }

        mse_previous_ = mse_;
    }
}
