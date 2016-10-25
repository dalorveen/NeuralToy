#include <gtest/gtest.h>

#include "layer.h"

TEST(LayerTest, neurons_number)
{
    neuralToy::Layer layer(3,
                           4,
                           neuralToy::TransferFunction::HYPERBOLIC_TANGENT);
    EXPECT_EQ(4, layer.neurons_number());

    ASSERT_DEATH_IF_SUPPORTED(
        neuralToy::Layer layer(3,
                               0,
                               neuralToy::TransferFunction::HYPERBOLIC_TANGENT),
        "neurons_number != 0");
}

TEST(LayerTest, get_transferFunction)
{
    neuralToy::Layer layer(3,
                           4,
                           neuralToy::TransferFunction::LOGISTIC);
    EXPECT_EQ(neuralToy::TransferFunction::LOGISTIC,
              layer.get_transferFunction());
}

TEST(LayerTest, set_transferFunction)
{
    neuralToy::Layer layer(3,
                           4,
                           neuralToy::TransferFunction::LOGISTIC);
    layer.set_transferFunction(neuralToy::TransferFunction::LINEAR);
    EXPECT_EQ(neuralToy::TransferFunction::LINEAR,
              layer.get_transferFunction());
}

TEST(LayerTest, get_neuron)
{
    neuralToy::Layer layer(5,
                           2,
                           neuralToy::TransferFunction::LOGISTIC);
    EXPECT_TRUE(typeid(layer.get_neuron(0)).hash_code()
                == typeid(neuralToy::Neuron).hash_code());
    EXPECT_TRUE(typeid(layer.get_neuron(1)).hash_code()
                == typeid(neuralToy::Neuron).hash_code());
    ASSERT_DEATH_IF_SUPPORTED(layer.get_neuron(2), "index < neurons_number_");
}

TEST(LayerTest, excite)
{
    neuralToy::Layer layer(4,
                           3,
                           neuralToy::TransferFunction::LOGISTIC);
    neuralToy::Pattern input(3);
    input[0] = 0.2;
    input[1] = 0.4;
    input[2] = 0.6;

    layer.adjust_neuron(0).set_weight(0, 2.4);
    layer.adjust_neuron(0).set_weight(1, -8.5);
    layer.adjust_neuron(0).set_weight(2, 2.1);
    layer.adjust_neuron(0).set_weight(3, -3.2);

    layer.adjust_neuron(1).set_weight(0, -9.7);
    layer.adjust_neuron(1).set_weight(1, -4.7);
    layer.adjust_neuron(1).set_weight(2, 11.8);
    layer.adjust_neuron(1).set_weight(3, -5.3);

    layer.adjust_neuron(2).set_weight(0, 9.7);
    layer.adjust_neuron(2).set_weight(1, 4.7);
    layer.adjust_neuron(2).set_weight(2, -1.8);
    layer.adjust_neuron(2).set_weight(3, -8.3);

    neuralToy::Pattern result(3);
    layer.excite(input, result);
    EXPECT_NEAR(0.007690876, result[0], 0.000000001);
    EXPECT_NEAR(0.115066732, result[1], 0.000000001);
    EXPECT_NEAR(0.00383402, result[2], 0.000000001);
}

TEST(LayerTest, copy_constructor)
{
    neuralToy::Layer layer(4,
                           3,
                           neuralToy::TransferFunction::LOGISTIC);

    layer.adjust_neuron(0).set_weight(0, 2.4);
    layer.adjust_neuron(0).set_weight(1, -8.5);
    layer.adjust_neuron(0).set_weight(2, 2.1);
    layer.adjust_neuron(0).set_weight(3, -3.2);

    layer.adjust_neuron(1).set_weight(0, -9.7);
    layer.adjust_neuron(1).set_weight(1, -4.7);
    layer.adjust_neuron(1).set_weight(2, 11.8);
    layer.adjust_neuron(1).set_weight(3, -5.3);

    layer.adjust_neuron(2).set_weight(0, 9.7);
    layer.adjust_neuron(2).set_weight(1, 4.7);
    layer.adjust_neuron(2).set_weight(2, -1.8);
    layer.adjust_neuron(2).set_weight(3, -8.3);

    neuralToy::Layer copy_layer(layer);
    EXPECT_EQ(neuralToy::TransferFunction::LOGISTIC,
              copy_layer.get_transferFunction());
    EXPECT_EQ(3, copy_layer.neurons_number());

    EXPECT_EQ(2.4, copy_layer.adjust_neuron(0).get_weight(0));
    EXPECT_EQ(-8.5, copy_layer.adjust_neuron(0).get_weight(1));
    EXPECT_EQ(2.1, copy_layer.adjust_neuron(0).get_weight(2));
    EXPECT_EQ(-3.2, copy_layer.adjust_neuron(0).get_weight(3));

    EXPECT_EQ(-9.7, copy_layer.adjust_neuron(1).get_weight(0));
    EXPECT_EQ(-4.7, copy_layer.adjust_neuron(1).get_weight(1));
    EXPECT_EQ(11.8, copy_layer.adjust_neuron(1).get_weight(2));
    EXPECT_EQ(-5.3, copy_layer.adjust_neuron(1).get_weight(3));

    EXPECT_EQ(9.7, copy_layer.adjust_neuron(2).get_weight(0));
    EXPECT_EQ(4.7, copy_layer.adjust_neuron(2).get_weight(1));
    EXPECT_EQ(-1.8, copy_layer.adjust_neuron(2).get_weight(2));
    EXPECT_EQ(-8.3, copy_layer.adjust_neuron(2).get_weight(3));
}

TEST(LayerTest, assignment_operator)
{
    neuralToy::Layer layer_1(4,
                             3,
                             neuralToy::TransferFunction::LOGISTIC);

    layer_1.adjust_neuron(0).set_weight(0, 2.4);
    layer_1.adjust_neuron(0).set_weight(1, -8.5);
    layer_1.adjust_neuron(0).set_weight(2, 2.1);
    layer_1.adjust_neuron(0).set_weight(3, -3.2);

    layer_1.adjust_neuron(1).set_weight(0, -9.7);
    layer_1.adjust_neuron(1).set_weight(1, -4.7);
    layer_1.adjust_neuron(1).set_weight(2, 11.8);
    layer_1.adjust_neuron(1).set_weight(3, -5.3);

    layer_1.adjust_neuron(2).set_weight(0, 9.7);
    layer_1.adjust_neuron(2).set_weight(1, 4.7);
    layer_1.adjust_neuron(2).set_weight(2, -1.8);
    layer_1.adjust_neuron(2).set_weight(3, -8.3);

    neuralToy::Layer layer_2(1,
                             1,
                             neuralToy::TransferFunction::LINEAR);
    layer_2.adjust_neuron(0).set_weight(0, 9.2);

    layer_2 = layer_1;

    EXPECT_EQ(neuralToy::TransferFunction::LOGISTIC,
              layer_2.get_transferFunction());
    EXPECT_EQ(3, layer_2.neurons_number());

    EXPECT_EQ(2.4, layer_2.adjust_neuron(0).get_weight(0));
    EXPECT_EQ(-8.5, layer_2.adjust_neuron(0).get_weight(1));
    EXPECT_EQ(2.1, layer_2.adjust_neuron(0).get_weight(2));
    EXPECT_EQ(-3.2, layer_2.adjust_neuron(0).get_weight(3));

    EXPECT_EQ(-9.7, layer_2.adjust_neuron(1).get_weight(0));
    EXPECT_EQ(-4.7, layer_2.adjust_neuron(1).get_weight(1));
    EXPECT_EQ(11.8, layer_2.adjust_neuron(1).get_weight(2));
    EXPECT_EQ(-5.3, layer_2.adjust_neuron(1).get_weight(3));

    EXPECT_EQ(9.7, layer_2.adjust_neuron(2).get_weight(0));
    EXPECT_EQ(4.7, layer_2.adjust_neuron(2).get_weight(1));
    EXPECT_EQ(-1.8, layer_2.adjust_neuron(2).get_weight(2));
    EXPECT_EQ(-8.3, layer_2.adjust_neuron(2).get_weight(3));
}
