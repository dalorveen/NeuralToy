#include "learningRule.h"

namespace neuralToy
{
    LearningRule::LearningRule(const Network& network, Learning learning)
    {
        learning_ = learning;
        network_   = new Network(network);
        errors_    = new double*[network.layers_number()];
        gradients_ = new double**[network.layers_number()];

        for (unsigned short l = 0; l < network.layers_number(); ++l) {
            errors_[l]    = new double[network.get_layer(l).neurons_number()];
            gradients_[l] = new double*[network.get_layer(l).neurons_number()];

            for (unsigned short n = 0;
                 n < network.get_layer(l).neurons_number(); ++n) {
                errors_[l][n] = 0;

                unsigned short wmax = network.get_layer(l).get_neuron(n)
                                      .inputs_number();
                gradients_[l][n]    = new double[wmax];

                for (unsigned short w = 0; w < wmax; ++w) {
                    gradients_[l][n][w] = 0;
                }
            }
        }
    }

    LearningRule::~LearningRule()
    {
        for (unsigned short l = 0; l < network_->layers_number(); ++l) {
            for (unsigned short n = 0;
                 n < network_->get_layer(l).neurons_number(); ++n) {
                delete [] gradients_[l][n];
                gradients_[l][n] = nullptr;
            }

            delete [] errors_[l];
            delete [] gradients_[l];
            errors_[l]    = nullptr;
            gradients_[l] = nullptr;
        }

        delete network_;
        delete [] errors_;
        delete [] gradients_;
        network_   = nullptr;
        errors_    = nullptr;
        gradients_ = nullptr;

    }

    Learning LearningRule::get_learning(void) const
    {
        return learning_;
    }

    double LearningRule::calculate_derivative(const double& value,
                                              TransferFunction tf)
    {
        switch (tf) {
		case TransferFunction::HYPERBOLIC_TANGENT:
			return 1.0 - std::pow(value, 2);

		case TransferFunction::LOGISTIC:
			return value * (1 - value);

        case TransferFunction::LINEAR:
			return 1;

        default:
            throw;
		}
    }

    double LearningRule::calculate_mean_squared_error(const DataSet& ds)
    {
        unsigned short lmax       = network_->layers_number() - 1;
        double total              = 0;
        double sum_squared_errors = 0;

        for (auto it = ds.cbegin(); it != ds.cend(); ++it) {
            network_->compute((**it).get_input());

            for (unsigned short j = 0; j < (**it).get_expected_output().size();
                 ++j) {
                sum_squared_errors += std::pow(
                                               (**it).get_expected_output()[j]
                                               - network_->get_output(lmax, j),
                                               2);
            }

            total += sum_squared_errors;
            sum_squared_errors = 0;
        }

        return total / ds.size();
    }

    Network LearningRule::get_network()
    {
        return *network_;
    }

    void LearningRule::calculate_errors_for_output_layer(const DataPair& dp)
    {
        unsigned short lmax = network_->layers_number() - 1;

        for (unsigned short n = 0;
             n < network_->get_layer(lmax).neurons_number(); ++n) {
            double out = network_->get_output(lmax, n);
            auto tf    = network_->get_layer(lmax).get_transferFunction();
            errors_[lmax][n] = calculate_derivative(out, tf)
                               * (out - dp.get_expected_output()[n]);
        }
    }

    void LearningRule::calculate_errors_for_hidden_layers()
    {
        for (unsigned short hl = network_->layers_number() - 2;
             hl != (unsigned short)(-1); --hl) {
            unsigned short next = hl + 1;

            for (unsigned short n = 0;
                 n < network_->get_layer(hl).neurons_number(); ++n) {
                double sum = 0;

                for (unsigned short m = 0;
                     m < network_->get_layer(next).neurons_number(); ++m) {
                    sum += network_->get_layer(next).get_neuron(m)
                           .get_weight(n) * errors_[next][m];
                }

                double out = network_->get_output(hl, n);
                auto tf    = network_->get_layer(hl).get_transferFunction();
                errors_[hl][n] = sum * calculate_derivative(out, tf);
            }
        }
    }

    void LearningRule::calculate_gradients(const DataPair& dp)
    {
        for (unsigned short l = network_->layers_number() - 1;
             l != (unsigned short)(-1); --l) {
            for (unsigned short n = 0;
                 n < network_->get_layer(l).neurons_number(); ++n) {
                if (l == 0) {
                    // input layer
                    for (unsigned short k = 0; k < dp.get_input().size(); ++k) {
                        gradients_[l][n][k]
                            = (learning_ == Learning::ONLINE ?
                               0 : gradients_[l][n][k])
                              + errors_[l][n] * dp.get_input()[k];
                    }

                    // bias
                    gradients_[l][n][dp.get_input().size()]
                        = (learning_ == Learning::ONLINE ?
                           0 : gradients_[l][n][dp.get_input().size()])
                          + errors_[l][n];
                }
                else {
                    // hidden layer
                    for (unsigned short k = 0;
                         k < network_->get_layer(l - 1).neurons_number(); ++k) {
                        gradients_[l][n][k]
                            = (learning_ == Learning::ONLINE ?
                               0 : gradients_[l][n][k])
                              + errors_[l][n] * network_->get_output(l - 1, k);
                    }

                    // bias
                    gradients_[l][n][network_->get_layer(l - 1)
                                     .neurons_number()]
                            = (learning_ == Learning::ONLINE ?
                               0 : gradients_[l][n][network_->get_layer(l - 1)
                                                    .neurons_number()])
                              + errors_[l][n];
                }
            }
        }
    }

    void LearningRule::calculate_backward_pass(const DataPair& dataPair)
    {
        network_->compute(dataPair.get_input());
        calculate_errors_for_output_layer(dataPair);
        calculate_errors_for_hidden_layers();
        calculate_gradients(dataPair);
    }

    const double& LearningRule::get_error(unsigned short layer_index,
                                          unsigned short neuron_index) const
    {
        return errors_[layer_index][neuron_index];
    }

    const double& LearningRule::get_gradient(unsigned short layer_index,
                                             unsigned short neuron_index,
                                             unsigned short weight_index) const
    {
        return gradients_[layer_index][neuron_index][weight_index];
    }

    void LearningRule::reset_gradients()
    {
        for (unsigned short l = 0; l < network_->layers_number(); ++l) {
            for (unsigned short n = 0;
                 n < network_->get_layer(l).neurons_number(); ++n) {
                unsigned short wmax = network_->get_layer(l).get_neuron(n)
                                      .inputs_number();

                for (unsigned short w = 0; w < wmax; ++w) {
                    gradients_[l][n][w] = 0;
                }
            }
        }
    }

    void LearningRule::accumulate_gradients(const DataSet& ds)
    {
        assert(learning_ == Learning::BATCH);

        reset_gradients();

        for (auto it = ds.cbegin(); it != ds.cend(); ++it) {
            calculate_backward_pass(**it);
        }
    }
}
