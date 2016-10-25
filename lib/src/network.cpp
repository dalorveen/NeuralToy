#include "network.h"

namespace neuralToy
{
    Network::Network(unsigned short inputs_number,
                     std::queue<unsigned short> hidden_layers_size,
                     unsigned short outputs_number)
    {
        inputs_number_ = inputs_number;
        layers_number_ = hidden_layers_size.size() + 1;
        layers_ = new Layer*[layers_number_];
        outputs_ = new Pattern*[layers_number_];
        unsigned short i = 0;

        while (!hidden_layers_size.empty()) {
            layers_[i] = new Layer(inputs_number + 1,
                                   hidden_layers_size.front(),
                                   TransferFunction::HYPERBOLIC_TANGENT);
            outputs_[i] = new Pattern(hidden_layers_size.front());
            i++;
            inputs_number = hidden_layers_size.front();
            hidden_layers_size.pop();
        }

        layers_[i] = new Layer(inputs_number + 1,
                               outputs_number,
                               TransferFunction::LINEAR);
        outputs_[i] = new Pattern(outputs_number);
    }

    Network::~Network()
    {
        for (unsigned short i = 0; i < layers_number_; ++i) {
            delete layers_[i];
            delete outputs_[i];
            layers_[i] = nullptr;
            outputs_[i] = nullptr;
        }

        delete [] layers_;
        delete [] outputs_;
        layers_ = nullptr;
        outputs_ = nullptr;
    }

    void Network::copy(const Network& other)
    {
        inputs_number_ = other.inputs_number_;
        layers_number_ = other.layers_number_;
        layers_ = new Layer*[other.layers_number_];
        outputs_ = new Pattern*[other.layers_number_];

        for (unsigned short i = 0; i < other.layers_number_; ++i) {
            layers_[i] = new Layer(*other.layers_[i]);
            outputs_[i] = new Pattern(*other.outputs_[i]);
        }
    }

    Network::Network(const Network& other)
    {
        copy(other);
    }

    Network& Network::operator=(const Network& rhs)
    {
        if (this == &rhs) return *this;

        this->~Network();
        copy(rhs);

        return *this;
    }

    const Layer& Network::get_layer(unsigned short layer_index) const
    {
        assert(layer_index < layers_number_);

        return *layers_[layer_index];
    }

    IAdjustment& Network::adjust_layer(unsigned short layer_index)
    {
        assert(layer_index < layers_number_);

        return *layers_[layer_index];
    }

    const Pattern& Network::compute(const Pattern& input)
    {
        for (unsigned short i = 0; i < layers_number_; ++i) {
            if (i == 0) {
                layers_[i]->excite(input, *outputs_[i]);
            } else {
                layers_[i]->excite(*outputs_[i - 1], *outputs_[i]);
            }
        }
        return *outputs_[layers_number_ - 1];
    }

    const double& Network::get_output(unsigned short layer_index,
                                      unsigned short neuron_index)
    {
        return outputs_[layer_index]->operator[](neuron_index);
    }

    const unsigned short Network::inputs_number(void) const
    {
        return inputs_number_;
    }

    const unsigned short Network::layers_number(void) const
    {
        return layers_number_;
    }

    /**
    ** Xavier Glorot
    ** Yoshua Bengio
    **/
    double Network::normalized_initialization(
                                            unsigned short size_previous_layer,
                                            unsigned short size_current_layer)
    {
        double a = std::sqrt( 6.0 / (size_previous_layer + size_current_layer) );
        std::uniform_real_distribution<double> distribution(-a, a);
        return distribution(generator_);
    }

    void Network::reset_weights(void)
    {
        generator_.seed( std::time(0) );
        for (unsigned short i = 0; i < layers_number_; ++i) {
            for (unsigned short j = 0; j < layers_[i]->neurons_number(); ++j) {
                for (unsigned short w = 0;
                     w < layers_[i]->get_neuron(j).inputs_number(); ++w) {
                    layers_[i]->adjust_neuron(j)
                                    .set_weight(
                        w,
                        normalized_initialization(
                                    layers_[i]->get_neuron(j).inputs_number(),
                                                layers_[i]->neurons_number()));
                }
            }
        }
    }
}
