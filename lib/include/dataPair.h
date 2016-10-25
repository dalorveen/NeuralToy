#ifndef DATAPAIR_H
#define DATAPAIR_H

#include "pattern.h"

namespace neuralToy
{
    class DataPair
    {
    public:
        DataPair(const Pattern&, const Pattern&);
        virtual ~DataPair();
        DataPair(const DataPair& other);
        DataPair& operator=(const DataPair& other);
        const Pattern& get_input(void) const;
        const Pattern& get_expected_output(void) const;

    protected:

    private:
        Pattern* input_;
        Pattern* expected_output_;
    };
}

#endif // DATAPAIR_H
