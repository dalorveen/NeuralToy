#ifndef IADJUSTMENT_H_INCLUDED
#define IADJUSTMENT_H_INCLUDED

#include "iTransferFunction.h"

namespace neuralToy
{
    class IAdjustment : public ITransferFunction
    {
    public:
        virtual IWeight& adjust_neuron(unsigned short) = 0;
    };
}

#endif // IADJUSTMENT_H_INCLUDED
