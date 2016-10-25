#include <gtest/gtest.h>
#include <typeinfo>

#include "network.h"

class NetworkTest : public ::testing::Test
{
protected:
    neuralToy::Network* network_;
    void SetUp()
    {
        std::queue<unsigned short> hidden_layers_size;
        hidden_layers_size.push(2);
        hidden_layers_size.push(4);
        network_ = new neuralToy::Network(4, hidden_layers_size, 3);

        network_->adjust_layer(0).adjust_neuron(0).set_weight(0, -0.225);
        network_->adjust_layer(0).adjust_neuron(0).set_weight(1, 0.112);
        network_->adjust_layer(0).adjust_neuron(0).set_weight(2, -0.011);
        network_->adjust_layer(0).adjust_neuron(0).set_weight(3, 0.323);
        network_->adjust_layer(0).adjust_neuron(0).set_weight(4, -0.053);

        network_->adjust_layer(0).adjust_neuron(1).set_weight(0, -0.363);
        network_->adjust_layer(0).adjust_neuron(1).set_weight(1, -0.027);
        network_->adjust_layer(0).adjust_neuron(1).set_weight(2, 0.149);
        network_->adjust_layer(0).adjust_neuron(1).set_weight(3, -0.389);
        network_->adjust_layer(0).adjust_neuron(1).set_weight(4, 0.49);

        network_->adjust_layer(1).adjust_neuron(0).set_weight(0, 0.185);
        network_->adjust_layer(1).adjust_neuron(0).set_weight(1, 0.473);
        network_->adjust_layer(1).adjust_neuron(0).set_weight(2, -0.261);

        network_->adjust_layer(1).adjust_neuron(1).set_weight(0, 0.367);
        network_->adjust_layer(1).adjust_neuron(1).set_weight(1, 0.054);
        network_->adjust_layer(1).adjust_neuron(1).set_weight(2, -0.372);

        network_->adjust_layer(1).adjust_neuron(2).set_weight(0, 0.354);
        network_->adjust_layer(1).adjust_neuron(2).set_weight(1, -0.112);
        network_->adjust_layer(1).adjust_neuron(2).set_weight(2, 0.333);

        network_->adjust_layer(1).adjust_neuron(3).set_weight(0, -0.43);
        network_->adjust_layer(1).adjust_neuron(3).set_weight(1, 0.448);
        network_->adjust_layer(1).adjust_neuron(3).set_weight(2, -0.094);

        network_->adjust_layer(2).adjust_neuron(0).set_weight(0, 0.284);
        network_->adjust_layer(2).adjust_neuron(0).set_weight(1, 0.217);
        network_->adjust_layer(2).adjust_neuron(0).set_weight(2, 0.35);
        network_->adjust_layer(2).adjust_neuron(0).set_weight(3, -0.106);
        network_->adjust_layer(2).adjust_neuron(0).set_weight(4, -0.395);

        network_->adjust_layer(2).adjust_neuron(1).set_weight(0, 0.291);
        network_->adjust_layer(2).adjust_neuron(1).set_weight(1, -0.227);
        network_->adjust_layer(2).adjust_neuron(1).set_weight(2, -0.402);
        network_->adjust_layer(2).adjust_neuron(1).set_weight(3, -0.323);
        network_->adjust_layer(2).adjust_neuron(1).set_weight(4, -0.491);

        network_->adjust_layer(2).adjust_neuron(2).set_weight(0, 0.418);
        network_->adjust_layer(2).adjust_neuron(2).set_weight(1, 0.231);
        network_->adjust_layer(2).adjust_neuron(2).set_weight(2, -0.1);
        network_->adjust_layer(2).adjust_neuron(2).set_weight(3, -0.448);
        network_->adjust_layer(2).adjust_neuron(2).set_weight(4, -0.016);
    }

    void TearDown()
    {
        delete network_;
    }
};

TEST_F(NetworkTest, get_layer)
{
    EXPECT_TRUE(typeid(network_->get_layer(0)).hash_code()
                == typeid(neuralToy::Layer).hash_code());
    EXPECT_TRUE(typeid(network_->get_layer(1)).hash_code()
                == typeid(neuralToy::Layer).hash_code());
    EXPECT_TRUE(typeid(network_->get_layer(2)).hash_code()
                == typeid(neuralToy::Layer).hash_code());
    ASSERT_DEATH_IF_SUPPORTED(network_->get_layer(3),
                              "layer_index < layers_number_");

    EXPECT_EQ(neuralToy::TransferFunction::HYPERBOLIC_TANGENT,
              network_->get_layer(0).get_transferFunction());
    EXPECT_EQ(neuralToy::TransferFunction::HYPERBOLIC_TANGENT,
              network_->get_layer(1).get_transferFunction());
    EXPECT_EQ(neuralToy::TransferFunction::LINEAR,
              network_->get_layer(2).get_transferFunction());
}

TEST_F(NetworkTest, adjust_layer)
{
    EXPECT_TRUE(typeid(network_->adjust_layer(0)).hash_code()
                == typeid(neuralToy::Layer).hash_code());
    EXPECT_TRUE(typeid(network_->adjust_layer(1)).hash_code()
                == typeid(neuralToy::Layer).hash_code());
    EXPECT_TRUE(typeid(network_->adjust_layer(2)).hash_code()
                == typeid(neuralToy::Layer).hash_code());
    ASSERT_DEATH_IF_SUPPORTED(network_->adjust_layer(3),
                              "layer_index < layers_number_");

    EXPECT_TRUE(typeid(network_->adjust_layer(0).adjust_neuron(0)).hash_code()
                == typeid(neuralToy::Neuron).hash_code());
    EXPECT_TRUE(typeid(network_->adjust_layer(0).adjust_neuron(1)).hash_code()
                == typeid(neuralToy::Neuron).hash_code());
    ASSERT_DEATH_IF_SUPPORTED(network_->adjust_layer(0).adjust_neuron(2),
                              "");

    EXPECT_EQ(neuralToy::TransferFunction::HYPERBOLIC_TANGENT,
              network_->adjust_layer(0).get_transferFunction());
    EXPECT_EQ(neuralToy::TransferFunction::HYPERBOLIC_TANGENT,
              network_->adjust_layer(1).get_transferFunction());
    EXPECT_EQ(neuralToy::TransferFunction::LINEAR,
              network_->adjust_layer(2).get_transferFunction());

    network_->adjust_layer(0).adjust_neuron(0).set_weight(0, 3.4);
    network_->adjust_layer(0).adjust_neuron(0).set_weight(1, 1.2);
    EXPECT_EQ(3.4, network_->adjust_layer(0).adjust_neuron(0).get_weight(0));
    EXPECT_EQ(1.2, network_->adjust_layer(0).adjust_neuron(0).get_weight(1));
}

TEST_F(NetworkTest, compute)
{
    neuralToy::Pattern input(4);
    input[0] = 0.2;
    input[1] = 0.4;
    input[2] = 0.6;
    input[3] = 0.8;

    const neuralToy::Pattern& result = network_->compute(input);
    EXPECT_NEAR(-0.35757, result[0], 0.00001);
    EXPECT_NEAR(-0.58234, result[1], 0.00001);
    EXPECT_NEAR(-0.13203, result[2], 0.00001);

    const neuralToy::Pattern& result2 = network_->compute(input);
    EXPECT_NEAR(-0.35757, result2[0], 0.00001);
    EXPECT_NEAR(-0.58234, result2[1], 0.00001);
    EXPECT_NEAR(-0.13203, result2[2], 0.00001);
}

TEST_F(NetworkTest, get_output)
{
    neuralToy::Pattern input(4);
    input[0] = 0.2;
    input[1] = 0.4;
    input[2] = 0.6;
    input[3] = 0.8;

    network_->compute(input);
    EXPECT_NEAR(0.19603, network_->get_output(0, 0), 0.00001);
    EXPECT_NEAR(0.18272, network_->get_output(0, 1), 0.00001);
    EXPECT_NEAR(-0.13743, network_->get_output(1, 0), 0.00001);
    EXPECT_NEAR(-0.28231, network_->get_output(1, 1), 0.00001);
    EXPECT_NEAR(0.36438, network_->get_output(1, 2), 0.00001);
    EXPECT_NEAR(-0.09613, network_->get_output(1, 3), 0.00001);
    EXPECT_NEAR(-0.35757, network_->get_output(2, 0), 0.00001);
    EXPECT_NEAR(-0.58234, network_->get_output(2, 1), 0.00001);
    EXPECT_NEAR(-0.13203, network_->get_output(2, 2), 0.00001);
}

TEST_F(NetworkTest, copy_constructor)
{
    network_->adjust_layer(0).set_transferFunction(
        neuralToy::TransferFunction::LOGISTIC);
    network_->adjust_layer(1).set_transferFunction(
        neuralToy::TransferFunction::LINEAR);
    network_->adjust_layer(2).set_transferFunction(
        neuralToy::TransferFunction::LOGISTIC);

    neuralToy::Network copy_network(*network_);

    EXPECT_EQ(neuralToy::TransferFunction::LOGISTIC,
              copy_network.adjust_layer(0).get_transferFunction());
    EXPECT_EQ(neuralToy::TransferFunction::LINEAR,
              copy_network.adjust_layer(1).get_transferFunction());
    EXPECT_EQ(neuralToy::TransferFunction::LOGISTIC,
              copy_network.adjust_layer(2).get_transferFunction());

    EXPECT_EQ(-0.225, copy_network.get_layer(0).get_neuron(0).get_weight(0));
    EXPECT_EQ(0.112, copy_network.get_layer(0).get_neuron(0).get_weight(1));
    EXPECT_EQ(-0.011, copy_network.get_layer(0).get_neuron(0).get_weight(2));
    EXPECT_EQ(0.323, copy_network.get_layer(0).get_neuron(0).get_weight(3));
    EXPECT_EQ(-0.053, copy_network.get_layer(0).get_neuron(0).get_weight(4));

    EXPECT_EQ(-0.363, copy_network.get_layer(0).get_neuron(1).get_weight(0));
    EXPECT_EQ(-0.027, copy_network.get_layer(0).get_neuron(1).get_weight(1));
    EXPECT_EQ(0.149, copy_network.get_layer(0).get_neuron(1).get_weight(2));
    EXPECT_EQ(-0.389, copy_network.get_layer(0).get_neuron(1).get_weight(3));
    EXPECT_EQ(0.49, copy_network.get_layer(0).get_neuron(1).get_weight(4));

    EXPECT_EQ(0.185, copy_network.get_layer(1).get_neuron(0).get_weight(0));
    EXPECT_EQ(0.473, copy_network.get_layer(1).get_neuron(0).get_weight(1));
    EXPECT_EQ(-0.261, copy_network.get_layer(1).get_neuron(0).get_weight(2));

    EXPECT_EQ(0.367, copy_network.get_layer(1).get_neuron(1).get_weight(0));
    EXPECT_EQ(0.054, copy_network.get_layer(1).get_neuron(1).get_weight(1));
    EXPECT_EQ(-0.372, copy_network.get_layer(1).get_neuron(1).get_weight(2));

    EXPECT_EQ(0.354, copy_network.get_layer(1).get_neuron(2).get_weight(0));
    EXPECT_EQ(-0.112, copy_network.get_layer(1).get_neuron(2).get_weight(1));
    EXPECT_EQ(0.333, copy_network.get_layer(1).get_neuron(2).get_weight(2));

    EXPECT_EQ(-0.43, copy_network.get_layer(1).get_neuron(3).get_weight(0));
    EXPECT_EQ(0.448, copy_network.get_layer(1).get_neuron(3).get_weight(1));
    EXPECT_EQ(-0.094, copy_network.get_layer(1).get_neuron(3).get_weight(2));

    EXPECT_EQ(0.284, copy_network.get_layer(2).get_neuron(0).get_weight(0));
    EXPECT_EQ(0.217, copy_network.get_layer(2).get_neuron(0).get_weight(1));
    EXPECT_EQ(0.35, copy_network.get_layer(2).get_neuron(0).get_weight(2));
    EXPECT_EQ(-0.106, copy_network.get_layer(2).get_neuron(0).get_weight(3));
    EXPECT_EQ(-0.395, copy_network.get_layer(2).get_neuron(0).get_weight(4));

    EXPECT_EQ(0.291, copy_network.get_layer(2).get_neuron(1).get_weight(0));
    EXPECT_EQ(-0.227, copy_network.get_layer(2).get_neuron(1).get_weight(1));
    EXPECT_EQ(-0.402, copy_network.get_layer(2).get_neuron(1).get_weight(2));
    EXPECT_EQ(-0.323, copy_network.get_layer(2).get_neuron(1).get_weight(3));
    EXPECT_EQ(-0.491, copy_network.get_layer(2).get_neuron(1).get_weight(4));

    EXPECT_EQ(0.418, copy_network.get_layer(2).get_neuron(2).get_weight(0));
    EXPECT_EQ(0.231, copy_network.get_layer(2).get_neuron(2).get_weight(1));
    EXPECT_EQ(-0.1, copy_network.get_layer(2).get_neuron(2).get_weight(2));
    EXPECT_EQ(-0.448, copy_network.get_layer(2).get_neuron(2).get_weight(3));
    EXPECT_EQ(-0.016, copy_network.get_layer(2).get_neuron(2).get_weight(4));
}

TEST_F(NetworkTest, assignment_operator)
{
    std::queue<unsigned short> hidden_layers_size;
    hidden_layers_size.push(1);
    neuralToy::Network n(1, hidden_layers_size, 1);
    n.adjust_layer(0).set_transferFunction(neuralToy::TransferFunction::LINEAR);
    n.adjust_layer(1).set_transferFunction(neuralToy::TransferFunction::LINEAR);
    n.adjust_layer(0).adjust_neuron(0).set_weight(0, -0.831);
    n.adjust_layer(0).adjust_neuron(0).set_weight(1, 5.721);
    n.adjust_layer(1).adjust_neuron(0).set_weight(0, 1.532);
    n.adjust_layer(1).adjust_neuron(0).set_weight(1, -8.245);

    *network_ = n;

    EXPECT_EQ(neuralToy::TransferFunction::LINEAR,
              network_->adjust_layer(0).get_transferFunction());
    EXPECT_EQ(neuralToy::TransferFunction::LINEAR,
              network_->adjust_layer(1).get_transferFunction());
    EXPECT_EQ(-0.831, network_->get_layer(0).get_neuron(0).get_weight(0));
    EXPECT_EQ(5.721, network_->get_layer(0).get_neuron(0).get_weight(1));
    EXPECT_EQ(1.532, network_->get_layer(1).get_neuron(0).get_weight(0));
    EXPECT_EQ(-8.245, network_->get_layer(1).get_neuron(0).get_weight(1));
}

TEST_F(NetworkTest, inputs_number)
{
    EXPECT_EQ(4, network_->inputs_number());
}

TEST_F(NetworkTest, layers_number)
{
    EXPECT_EQ(3, network_->layers_number());
}

TEST_F(NetworkTest, DISABLED_reset_weights)
{
    network_->reset_weights();

    EXPECT_NE(-0.225, network_->get_layer(0).get_neuron(0).get_weight(0));
    EXPECT_NE(0.112, network_->get_layer(0).get_neuron(0).get_weight(1));
    EXPECT_NE(-0.011, network_->get_layer(0).get_neuron(0).get_weight(2));
    EXPECT_NE(0.323, network_->get_layer(0).get_neuron(0).get_weight(3));
    EXPECT_EQ(0, network_->get_layer(0).get_neuron(0).get_weight(4));

    EXPECT_NE(-0.363, network_->get_layer(0).get_neuron(1).get_weight(0));
    EXPECT_NE(-0.027, network_->get_layer(0).get_neuron(1).get_weight(1));
    EXPECT_NE(0.149, network_->get_layer(0).get_neuron(1).get_weight(2));
    EXPECT_NE(-0.389, network_->get_layer(0).get_neuron(1).get_weight(3));
    EXPECT_EQ(0, network_->get_layer(0).get_neuron(1).get_weight(4));

    EXPECT_NE(0.185, network_->get_layer(1).get_neuron(0).get_weight(0));
    EXPECT_NE(0.473, network_->get_layer(1).get_neuron(0).get_weight(1));
    EXPECT_EQ(0, network_->get_layer(1).get_neuron(0).get_weight(2));

    EXPECT_NE(0.367, network_->get_layer(1).get_neuron(1).get_weight(0));
    EXPECT_NE(0.054, network_->get_layer(1).get_neuron(1).get_weight(1));
    EXPECT_EQ(0, network_->get_layer(1).get_neuron(1).get_weight(2));

    EXPECT_NE(0.354, network_->get_layer(1).get_neuron(2).get_weight(0));
    EXPECT_NE(-0.112, network_->get_layer(1).get_neuron(2).get_weight(1));
    EXPECT_EQ(0, network_->get_layer(1).get_neuron(2).get_weight(2));

    EXPECT_NE(-0.43, network_->get_layer(1).get_neuron(3).get_weight(0));
    EXPECT_NE(0.448, network_->get_layer(1).get_neuron(3).get_weight(1));
    EXPECT_EQ(0, network_->get_layer(1).get_neuron(3).get_weight(2));

    EXPECT_NE(0.284, network_->get_layer(2).get_neuron(0).get_weight(0));
    EXPECT_NE(0.217, network_->get_layer(2).get_neuron(0).get_weight(1));
    EXPECT_NE(0.35, network_->get_layer(2).get_neuron(0).get_weight(2));
    EXPECT_NE(-0.106, network_->get_layer(2).get_neuron(0).get_weight(3));
    EXPECT_EQ(0, network_->get_layer(2).get_neuron(0).get_weight(4));

    EXPECT_NE(0.291, network_->get_layer(2).get_neuron(1).get_weight(0));
    EXPECT_NE(-0.227, network_->get_layer(2).get_neuron(1).get_weight(1));
    EXPECT_NE(-0.402, network_->get_layer(2).get_neuron(1).get_weight(2));
    EXPECT_NE(-0.323, network_->get_layer(2).get_neuron(1).get_weight(3));
    EXPECT_EQ(0, network_->get_layer(2).get_neuron(1).get_weight(4));

    EXPECT_NE(0.418, network_->get_layer(2).get_neuron(2).get_weight(0));
    EXPECT_NE(0.231, network_->get_layer(2).get_neuron(2).get_weight(1));
    EXPECT_NE(-0.1, network_->get_layer(2).get_neuron(2).get_weight(2));
    EXPECT_NE(-0.448, network_->get_layer(2).get_neuron(2).get_weight(3));
    EXPECT_EQ(0, network_->get_layer(2).get_neuron(2).get_weight(4));
}
