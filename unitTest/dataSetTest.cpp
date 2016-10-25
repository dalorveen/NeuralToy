#include <gtest/gtest.h>

#include "dataSet.h"

TEST(DataSet, iterator)
{
    neuralToy::Pattern input_1(3);
    input_1[0] = 1.2;
    input_1[1] = 3.4;
    input_1[2] = 5.6;
    neuralToy::Pattern expected_output_1(2);
    expected_output_1[0] = 7.8;
    expected_output_1[1] = 9.1;

    neuralToy::Pattern input_2(3);
    input_2[0] = 11.12;
    input_2[1] = 13.14;
    input_2[2] = 15.16;
    neuralToy::Pattern expected_output_2(2);
    expected_output_2[0] = 17.18;
    expected_output_2[1] = 19.2;

    neuralToy::Pattern input_3(3);
    input_3[0] = 21.22;
    input_3[1] = 23.24;
    input_3[2] = 25.26;
    neuralToy::Pattern expected_output_3(2);
    expected_output_3[0] = 27.28;
    expected_output_3[1] = 29.3;

    neuralToy::DataPair dataPair_1(input_1, expected_output_1);
    neuralToy::DataPair dataPair_2(input_2, expected_output_2);
    neuralToy::DataPair dataPair_3(input_3, expected_output_3);

    neuralToy::DataSet dataSet;
    dataSet.add(dataPair_1);
    dataSet.add(dataPair_2);
    dataSet.add(dataPair_3);

    auto it = dataSet.cbegin();
    EXPECT_EQ(21.22, (*it)->get_input()[0]);
    EXPECT_EQ(23.24, (*it)->get_input()[1]);
    EXPECT_EQ(25.26, (*it)->get_input()[2]);
    EXPECT_EQ(27.28, (*it)->get_expected_output()[0]);
    EXPECT_EQ(29.3, (*it)->get_expected_output()[1]);
    ++it;
    EXPECT_EQ(11.12, (*it)->get_input()[0]);
    EXPECT_EQ(13.14, (*it)->get_input()[1]);
    EXPECT_EQ(15.16, (*it)->get_input()[2]);
    EXPECT_EQ(17.18, (*it)->get_expected_output()[0]);
    EXPECT_EQ(19.2, (*it)->get_expected_output()[1]);
    ++it;
    EXPECT_EQ(1.2, (*it)->get_input()[0]);
    EXPECT_EQ(3.4, (*it)->get_input()[1]);
    EXPECT_EQ(5.6, (*it)->get_input()[2]);
    EXPECT_EQ(7.8, (*it)->get_expected_output()[0]);
    EXPECT_EQ(9.1, (*it)->get_expected_output()[1]);
    ++it;
    EXPECT_EQ(it, dataSet.cend());
}

TEST(DataSet, size)
{
    neuralToy::Pattern input(4);
    neuralToy::Pattern expected_output(2);
    neuralToy::DataPair dataPair(input, expected_output);

    neuralToy::DataSet dataSet;
    EXPECT_EQ(0, dataSet.size());
    dataSet.add(dataPair);
    EXPECT_EQ(1, dataSet.size());
    dataSet.add(dataPair);
    dataSet.add(dataPair);
    dataSet.add(dataPair);
    dataSet.add(dataPair);
    EXPECT_EQ(5, dataSet.size());
}

TEST(DataSet, copy_constructor)
{
    neuralToy::Pattern input_1(3);
    input_1[0] = 1.2;
    input_1[1] = 3.4;
    input_1[2] = 5.6;
    neuralToy::Pattern expected_output_1(2);
    expected_output_1[0] = 7.8;
    expected_output_1[1] = 9.1;

    neuralToy::Pattern input_2(3);
    input_2[0] = 11.12;
    input_2[1] = 13.14;
    input_2[2] = 15.16;
    neuralToy::Pattern expected_output_2(2);
    expected_output_2[0] = 17.18;
    expected_output_2[1] = 19.2;

    neuralToy::Pattern input_3(3);
    input_3[0] = 21.22;
    input_3[1] = 23.24;
    input_3[2] = 25.26;
    neuralToy::Pattern expected_output_3(2);
    expected_output_3[0] = 27.28;
    expected_output_3[1] = 29.3;

    neuralToy::DataPair dataPair_1(input_1, expected_output_1);
    neuralToy::DataPair dataPair_2(input_2, expected_output_2);
    neuralToy::DataPair dataPair_3(input_3, expected_output_3);

    neuralToy::DataSet dataSet;
    dataSet.add(dataPair_1);
    dataSet.add(dataPair_2);
    dataSet.add(dataPair_3);

    neuralToy::DataSet copy_dataSet(dataSet);
    auto it = copy_dataSet.cbegin();
    EXPECT_EQ(21.22, (*it)->get_input()[0]);
    EXPECT_EQ(23.24, (*it)->get_input()[1]);
    EXPECT_EQ(25.26, (*it)->get_input()[2]);
    EXPECT_EQ(27.28, (*it)->get_expected_output()[0]);
    EXPECT_EQ(29.3, (*it)->get_expected_output()[1]);
    ++it;
    EXPECT_EQ(11.12, (*it)->get_input()[0]);
    EXPECT_EQ(13.14, (*it)->get_input()[1]);
    EXPECT_EQ(15.16, (*it)->get_input()[2]);
    EXPECT_EQ(17.18, (*it)->get_expected_output()[0]);
    EXPECT_EQ(19.2, (*it)->get_expected_output()[1]);
    ++it;
    EXPECT_EQ(1.2, (*it)->get_input()[0]);
    EXPECT_EQ(3.4, (*it)->get_input()[1]);
    EXPECT_EQ(5.6, (*it)->get_input()[2]);
    EXPECT_EQ(7.8, (*it)->get_expected_output()[0]);
    EXPECT_EQ(9.1, (*it)->get_expected_output()[1]);
}
