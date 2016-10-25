#ifndef LAYER_H
#define LAYER_H

#include <cassert>

#include "iAdjustment.h"
#include "neuron.h"


namespace neuralToy
{
    class Layer : public IAdjustment
    {
    public:
        Layer(unsigned short, unsigned short, TransferFunction);
        virtual ~Layer();
        Layer(const Layer& other);
        Layer& operator=(const Layer& other);
        unsigned short neurons_number() const;
        TransferFunction get_transferFunction(void) const;
        void set_transferFunction(TransferFunction tf);
        const Neuron& get_neuron(unsigned short) const;
        IWeight& adjust_neuron(unsigned short);
        void excite(const Pattern&, Pattern&);

    protected:

    private:
        unsigned short neurons_number_;
        TransferFunction tf_;
        Neuron** neurons_;
        void copy(const Layer&);
    };
}

#endif // LAYER_H
