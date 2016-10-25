#include "dataPair.h"

namespace neuralToy
{
    DataPair::DataPair(const Pattern& input, const Pattern& expected_output)
    {
        input_ = new Pattern(input);
        expected_output_ = new Pattern(expected_output);
    }

    DataPair::~DataPair()
    {
        delete input_;
        delete expected_output_;
        input_ = nullptr;
        expected_output_ = nullptr;
    }

    DataPair::DataPair(const DataPair& other)
    {
        input_ = new Pattern(*other.input_);
        expected_output_ = new Pattern(*other.expected_output_);
    }

    DataPair& DataPair::operator=(const DataPair& rhs)
    {
        if (this == &rhs) return *this; // handle self assignment
        //assignment operator
        return *this;
    }

    const Pattern& DataPair::get_input(void) const
    {
        return *input_;
    }

    const Pattern& DataPair::get_expected_output(void) const
    {
        return *expected_output_;
    }
}
