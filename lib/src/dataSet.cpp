#include "dataSet.h"

namespace neuralToy
{
    DataSet::DataSet()
    {
        size_ = 0;
    }

    DataSet::~DataSet()
    {
        for (auto it = dataPairs_.begin(); it != dataPairs_.end(); ++it) {
            delete *it;
        }
    }

    DataSet::DataSet(const DataSet& other)
    {
        size_ = other.size_;
        for (auto it = other.cbegin(); it != other.cend(); ++it) {
            dataPairs_.push_front(new DataPair(**it));
        }
        dataPairs_.reverse();
    }

    DataSet& DataSet::operator=(const DataSet& rhs)
    {
        if (this == &rhs) return *this; // handle self assignment
        //assignment operator
        return *this;
    }

    std::size_t DataSet::size(void) const
    {
        return size_;
    }

    void DataSet::add(const DataPair& dp)
    {
        dataPairs_.push_front(new DataPair(dp));
        ++size_;
    }

    const std::forward_list<DataPair*>::const_iterator DataSet::cbegin()
                                                                const noexcept
    {
        return dataPairs_.cbegin();
    }

    const std::forward_list<DataPair*>::const_iterator DataSet::cend()
                                                                const noexcept
    {
        return dataPairs_.cend();
    }
}
