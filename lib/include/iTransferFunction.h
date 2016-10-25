#ifndef ITRANSFERFUNCTION_H_INCLUDED
#define ITRANSFERFUNCTION_H_INCLUDED

#include "neuron.h"

namespace neuralToy
{
    class ITransferFunction
    {
    public:
        virtual TransferFunction get_transferFunction(void) const = 0;
        virtual void set_transferFunction(TransferFunction tf) = 0;
    };
}

#endif // ITRANSFERFUNCTION_H_INCLUDED
