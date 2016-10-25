#ifndef DATASET_H
#define DATASET_H

#include <forward_list>
#include <cstddef>

#include "dataPair.h"

namespace neuralToy
{
    class DataSet
    {
    public:
        DataSet();
        virtual ~DataSet();
        DataSet(const DataSet& other);
        DataSet& operator=(const DataSet& other);
        std::size_t size(void) const;
        void add(const DataPair&);
        const std::forward_list<DataPair*>::const_iterator cbegin()
                                                                const noexcept;
        const std::forward_list<DataPair*>::const_iterator cend()
                                                                const noexcept;

    protected:

    private:
        std::size_t size_;
        std::forward_list<DataPair*> dataPairs_;
    };
}

#endif // DATASET_H
