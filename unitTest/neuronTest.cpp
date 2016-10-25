#include <gtest/gtest.h>

#include "neuron.h"

TEST(NeuronTest, get_inputs_number)
{
    neuralToy::Neuron neuron(3);
    EXPECT_EQ(3, neuron.inputs_number());

    ASSERT_DEATH_IF_SUPPORTED(neuralToy::Neuron neuron(0),
                              "inputs_number != 0");
}

TEST(NeuronTest, get_weight)
{
    neuralToy::Neuron neuron(3);
    EXPECT_EQ(0, neuron.get_weight(0));
    EXPECT_EQ(0, neuron.get_weight(1));
    EXPECT_EQ(0, neuron.get_weight(2));
    ASSERT_DEATH_IF_SUPPORTED(neuron.get_weight(3), "index < inputs_number_");
}

TEST(NeuronTest, set_weight)
{
    neuralToy::Neuron neuron(3);
    neuron.set_weight(0, 1.2);
    neuron.set_weight(1, 3.4);
    neuron.set_weight(2, 5.6);
    ASSERT_DEATH_IF_SUPPORTED(neuron.set_weight(3, 7.8),
                              "index < inputs_number_");
}

TEST(NeuronTest, fire)
{
    neuralToy::Pattern input(3);
    input[0] = 0.2;
    input[1] = 0.4;
    input[2] = 0.6;

    neuralToy::Neuron neuron(4);
    neuron.set_weight(0, 1.1);
    neuron.set_weight(1, -2.2);
    neuron.set_weight(2, 3.3);
    neuron.set_weight(3, -4.4); // bias

    EXPECT_NEAR(-0.995784398,
                neuron.fire(input,
                            neuralToy::TransferFunction::HYPERBOLIC_TANGENT),
                0.000000001);
    EXPECT_NEAR(0.043939815,
                neuron.fire(input,
                            neuralToy::TransferFunction::LOGISTIC),
                0.000000001);
    EXPECT_NEAR(-3.080000000,
                neuron.fire(input,
                            neuralToy::TransferFunction::LINEAR),
                0.000000001);
}

TEST(NeuronTest, copy_constructor)
{
    neuralToy::Neuron neuron(4);
    neuron.set_weight(0, 1.2);
    neuron.set_weight(1, 3.4);
    neuron.set_weight(2, 5.6);
    neuron.set_weight(3, 7.8);

    neuralToy::Neuron copy_neuron(neuron);
    EXPECT_EQ(1.2, copy_neuron.get_weight(0));
    EXPECT_EQ(3.4, copy_neuron.get_weight(1));
    EXPECT_EQ(5.6, copy_neuron.get_weight(2));
    EXPECT_EQ(7.8, copy_neuron.get_weight(3));
    ASSERT_DEATH_IF_SUPPORTED(copy_neuron.get_weight(4),
                              "index < inputs_number_");
}

TEST(NeuronTest, assignment_operator)
{
    neuralToy::Neuron neuron_1(4);
    neuron_1.set_weight(0, 1.2);
    neuron_1.set_weight(1, 3.4);
    neuron_1.set_weight(2, 5.6);
    neuron_1.set_weight(3, 7.8);

    neuralToy::Neuron neuron_2(1);
    neuron_2.set_weight(0, 9.4);

    neuron_2 = neuron_1;

    EXPECT_EQ(1.2, neuron_2.get_weight(0));
    EXPECT_EQ(3.4, neuron_2.get_weight(1));
    EXPECT_EQ(5.6, neuron_2.get_weight(2));
    EXPECT_EQ(7.8, neuron_2.get_weight(3));
}
