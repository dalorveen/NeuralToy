#include "neuron.h"

namespace neuralToy
{
    Neuron::Neuron(unsigned short inputs_number)
    {
        assert(inputs_number != 0);

        inputs_number_ = inputs_number;
        weights_ = new double[inputs_number]{0};
    }

    Neuron::~Neuron()
    {
        delete [] weights_;
        weights_ = nullptr;
    }

    void Neuron::copy(const Neuron& other)
    {
        assert(other.inputs_number_ != 0);

        inputs_number_ = 0;
        weights_ = new double[other.inputs_number_];

        do {
            weights_[inputs_number_] = other.weights_[inputs_number_];
            inputs_number_++;
        } while (inputs_number_ != other.inputs_number_);
    }

    Neuron::Neuron(const Neuron& other)
    {
        copy(other);
    }

    Neuron& Neuron::operator=(const Neuron& rhs)
    {
        if (this == &rhs) return *this;

        delete [] weights_;
        copy(rhs);

        return *this;
    }

    unsigned short Neuron::inputs_number() const
    {
        return inputs_number_;
    }

    const double& Neuron::get_weight(unsigned short index) const
    {
        assert(index < inputs_number_);

        return weights_[index];
    }

    void Neuron::set_weight(unsigned short index, const double& value)
    {
        assert(index < inputs_number_);

        weights_[index] = value;
    }

    double Neuron::fire(const Pattern& pattern, TransferFunction tf)
    {
		assert(pattern.size() == inputs_number_ - 1);

        double sum = 0;

        for (unsigned short i = 0; i < pattern.size(); ++i) {
            sum += pattern[i] * get_weight(i);
        }

        sum += 1.0 * get_weight(inputs_number_ - 1);

        /*
        if (isinf(sum)) {
            if (sum < 0) {
                sum = DBL_MIN;
            } else {
                sum = DBL_MAX;
            }
        }
        */

        switch (tf) {
		case TransferFunction::HYPERBOLIC_TANGENT:
			return std::tanh(sum);

		case TransferFunction::LOGISTIC:
			return 1.0 / (1.0 + std::exp(-1.0 * sum));

        case TransferFunction::LINEAR:
			return sum;

        default:
            throw;
		}
    }
}
