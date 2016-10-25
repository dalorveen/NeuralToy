#ifndef IWEIGHT_H_INCLUDED
#define IWEIGHT_H_INCLUDED

namespace neuralToy
{
    class IWeight
    {
    public:
        virtual const double& get_weight(unsigned short) const = 0;
        virtual void set_weight(unsigned short, const double&) = 0;
    };
}

#endif // IWEIGHT_H_INCLUDED
