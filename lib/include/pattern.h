#ifndef PATTERN_H
#define PATTERN_H

#include <cassert>

namespace neuralToy
{
    class Pattern
    {
    public:
        Pattern(unsigned short);
        virtual ~Pattern();
        Pattern(const Pattern& other);
        Pattern& operator=(const Pattern& other);
        double& operator[](unsigned short index);
        const double& operator[](unsigned short index) const;
        unsigned short size(void) const;

    protected:

    private:
        unsigned short size_;
        double* data_;
    };
}

#endif // PATTERN_H
