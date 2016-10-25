#ifndef NEURON_H
#define NEURON_H

#include <cfloat>
#include <cassert>
#include <cmath>
#include <list>

#include "iWeight.h"
#include "pattern.h"


namespace neuralToy
{
    enum TransferFunction
    {
        HYPERBOLIC_TANGENT,
        LOGISTIC,
        LINEAR
    };

    class Neuron : public IWeight
    {
    public:
        Neuron(unsigned short);
        virtual ~Neuron();
        Neuron(const Neuron& other);
        Neuron& operator=(const Neuron& other);
        unsigned short inputs_number(void) const;
        const double& get_weight(unsigned short) const;
        void set_weight(unsigned short, const double&);
        double fire(const Pattern&, TransferFunction);

    protected:

    private:
        unsigned short inputs_number_;
        double* weights_;
        void copy(const Neuron&);
    };
}

#endif // NEURON_H
