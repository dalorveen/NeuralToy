#ifndef NETWORK_H
#define NETWORK_H

#include <cstdlib>
#include <ctime>
#include <queue>
#include <random>

#include "layer.h"

namespace neuralToy
{
    class Network
    {
    public:
        Network(unsigned short, std::queue<unsigned short>, unsigned short);
        virtual ~Network();
        Network(const Network& other);
        Network& operator=(const Network& other);
        const Layer& get_layer(unsigned short) const;
        IAdjustment& adjust_layer(unsigned short);
        const Pattern& compute(const Pattern&);
        const double& get_output(unsigned short, unsigned short);
        const unsigned short inputs_number(void) const;
        const unsigned short layers_number(void) const;
        void reset_weights(void);

    protected:

    private:
        unsigned short inputs_number_;
        unsigned short layers_number_;
        Layer** layers_;
        Pattern** outputs_;
        double normalized_initialization(unsigned short, unsigned short);
        std::default_random_engine generator_;
        void copy(const Network&);
    };
}

#endif // NETWORK_H
