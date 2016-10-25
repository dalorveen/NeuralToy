#include "layer.h"

namespace neuralToy
{
    Layer::Layer(unsigned short inputs_number,
                 unsigned short neurons_number,
                 TransferFunction tf)
    {
        assert(inputs_number != 0);
        assert(neurons_number != 0);

        neurons_number_ = neurons_number;
        tf_ = tf;
        neurons_ = new Neuron*[neurons_number];

        for (unsigned short i = 0; i < neurons_number; i++) {
			neurons_[i] = new Neuron(inputs_number);
		}
    }

    Layer::~Layer()
    {
        for (unsigned short i = 0; i < neurons_number_; ++i) {
            delete neurons_[i];
            neurons_[i] = nullptr;
        }

        delete [] neurons_;
        neurons_ = nullptr;
    }

    void Layer::copy(const Layer& other)
    {
        neurons_number_ = other.neurons_number_;
        tf_ = other.tf_;
        neurons_ = new Neuron*[other.neurons_number_];

        for (unsigned short i = 0; i < other.neurons_number_; i++) {
			neurons_[i] = new Neuron(*other.neurons_[i]);
		}
    }

    Layer::Layer(const Layer& other)
    {
        copy(other);
    }

    Layer& Layer::operator=(const Layer& rhs)
    {
        if (this == &rhs) return *this;

        this->~Layer();
        copy(rhs);

        return *this;
    }

    unsigned short Layer::neurons_number() const
    {
        return neurons_number_;
    }

    TransferFunction Layer::get_transferFunction() const
    {
        return tf_;
    }

    void Layer::set_transferFunction(TransferFunction tf)
    {
        tf_ = tf;
    }

    const Neuron& Layer::get_neuron(unsigned short index) const
    {
        assert(index < neurons_number_);

        return *neurons_[index];
    }

    IWeight& Layer::adjust_neuron(unsigned short index)
    {
        assert(index < neurons_number_);

        return *neurons_[index];
    }

    void Layer::excite(const Pattern& input, Pattern& output)
    {
        for (unsigned short i = 0; i < neurons_number_; ++i) {
            output.operator[](i) = neurons_[i]->fire(input, tf_);
        }
    }
}
