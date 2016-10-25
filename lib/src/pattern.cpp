#include "pattern.h"

namespace neuralToy
{
    Pattern::Pattern(unsigned short size)
    {
        assert(size != 0);
        size_ = size;
        data_ = new double[size]{0};
    }

    Pattern::~Pattern()
    {
        delete [] data_;
        data_ = nullptr;
    }

    Pattern::Pattern(const Pattern& other)
    {
        size_ = 0;
        data_ = new double[other.size_];
        do {
            data_[size_] = other.data_[size_];
            size_++;
        } while (size_ != other.size_);
    }

    Pattern& Pattern::operator=(const Pattern& rhs)
    {
        if (this == &rhs) return *this; // handle self assignment
        size_ = 0;
        this->~Pattern();
        data_ = new double[rhs.size_];
        do {
            data_[size_] = rhs.data_[size_];
            size_++;
        } while (size_ != rhs.size_);
        return *this;
    }

    double& Pattern::operator[](unsigned short index)
    {
        return data_[index];
    }

    const double& Pattern::operator[](unsigned short index) const
    {
        return data_[index];
    }

    unsigned short Pattern::size(void) const
    {
        return size_;
    }
}
