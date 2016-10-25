#include <gtest/gtest.h>

#include "gradientDescent.h"

TEST(GradientDescentTest, train_LOGISTIC)
{
    std::queue<unsigned short> hidden_layers_size;
    hidden_layers_size.push(2);
    neuralToy::Network network(2, hidden_layers_size, 1);

    network.adjust_layer(0).set_transferFunction(
                                        neuralToy::TransferFunction::LOGISTIC);
    network.adjust_layer(1).set_transferFunction(
                                        neuralToy::TransferFunction::LOGISTIC);

    network.adjust_layer(0).adjust_neuron(0).set_weight(0, 0.15);
    network.adjust_layer(0).adjust_neuron(0).set_weight(1, 0.2);
    network.adjust_layer(0).adjust_neuron(0).set_weight(2, 0.35);

    network.adjust_layer(0).adjust_neuron(1).set_weight(0, 0.25);
    network.adjust_layer(0).adjust_neuron(1).set_weight(1, 0.3);
    network.adjust_layer(0).adjust_neuron(1).set_weight(2, 0.35);

    network.adjust_layer(1).adjust_neuron(0).set_weight(0, 0.4);
    network.adjust_layer(1).adjust_neuron(0).set_weight(1, 0.45);
    network.adjust_layer(1).adjust_neuron(0).set_weight(2, 0.5);

    neuralToy::GradientDescent gd(network, neuralToy::Learning::ONLINE);

    neuralToy::Pattern input_1(2);
    input_1[0] = 0;
    input_1[1] = 0;
    neuralToy::Pattern target_1(1);
    target_1[0] = 0;

    neuralToy::Pattern input_2(2);
    input_2[0] = 0;
    input_2[1] = 1;
    neuralToy::Pattern target_2(1);
    target_2[0] = 1;

    neuralToy::Pattern input_3(2);
    input_3[0] = 1;
    input_3[1] = 0;
    neuralToy::Pattern target_3(1);
    target_3[0] = 1;

    neuralToy::Pattern input_4(2);
    input_4[0] = 1;
    input_4[1] = 1;
    neuralToy::Pattern target_4(1);
    target_4[0] = 0;

    neuralToy::DataPair dp_1(input_1, target_1);
    neuralToy::DataPair dp_2(input_2, target_2);
    neuralToy::DataPair dp_3(input_3, target_3);
    neuralToy::DataPair dp_4(input_4, target_4);

    neuralToy::DataSet ds;
    ds.add(dp_4);
    ds.add(dp_3);
    ds.add(dp_2);
    ds.add(dp_1);

    for (int i = 0; i < 76; ++i) {
        gd.train(ds);
        switch (i) {
        case 0:
            EXPECT_NEAR(0.293059275, gd.calculate_mean_squared_error(ds), 0.00000001);
            break;
        case 1:
            EXPECT_NEAR(0.281230754, gd.calculate_mean_squared_error(ds), 0.00000001);
            break;
        case 2:
            EXPECT_NEAR(0.271940259, gd.calculate_mean_squared_error(ds), 0.00000001);
            break;
        case 75:
            EXPECT_NEAR(0.249992677, gd.calculate_mean_squared_error(ds), 0.00000001);
            break;
        }
    }

    neuralToy::Network n(gd.get_network());
    n.compute(input_1);
    EXPECT_NEAR(0.497197018, n.get_output(1, 0), 0.00000001);
    n.compute(input_2);
    EXPECT_NEAR(0.498388197, n.get_output(1, 0), 0.00000001);
    n.compute(input_3);
    EXPECT_NEAR(0.498321502, n.get_output(1, 0), 0.00000001);
    n.compute(input_4);
    EXPECT_NEAR(0.499469834, n.get_output(1, 0), 0.00000001);
}

TEST(GradientDescentTest, train_HYPERBOLIC_TANGENT)
{
    std::queue<unsigned short> hidden_layers_size;
    hidden_layers_size.push(2);
    neuralToy::Network network(2, hidden_layers_size, 1);

    network.adjust_layer(0).set_transferFunction(
                                        neuralToy::TransferFunction::HYPERBOLIC_TANGENT);
    network.adjust_layer(1).set_transferFunction(
                                        neuralToy::TransferFunction::HYPERBOLIC_TANGENT);

    network.adjust_layer(0).adjust_neuron(0).set_weight(0, 0.15);
    network.adjust_layer(0).adjust_neuron(0).set_weight(1, 0.2);
    network.adjust_layer(0).adjust_neuron(0).set_weight(2, 0.35);

    network.adjust_layer(0).adjust_neuron(1).set_weight(0, 0.25);
    network.adjust_layer(0).adjust_neuron(1).set_weight(1, 0.3);
    network.adjust_layer(0).adjust_neuron(1).set_weight(2, 0.35);

    network.adjust_layer(1).adjust_neuron(0).set_weight(0, 0.4);
    network.adjust_layer(1).adjust_neuron(0).set_weight(1, 0.45);
    network.adjust_layer(1).adjust_neuron(0).set_weight(2, 0.5);

    neuralToy::GradientDescent gd(network, neuralToy::Learning::ONLINE);

    neuralToy::Pattern input_1(2);
    input_1[0] = 0;
    input_1[1] = 0;
    neuralToy::Pattern target_1(1);
    target_1[0] = 0;

    neuralToy::Pattern input_2(2);
    input_2[0] = 0;
    input_2[1] = 1;
    neuralToy::Pattern target_2(1);
    target_2[0] = 1;

    neuralToy::Pattern input_3(2);
    input_3[0] = 1;
    input_3[1] = 0;
    neuralToy::Pattern target_3(1);
    target_3[0] = 1;

    neuralToy::Pattern input_4(2);
    input_4[0] = 1;
    input_4[1] = 1;
    neuralToy::Pattern target_4(1);
    target_4[0] = 0;

    neuralToy::DataPair dp_1(input_1, target_1);
    neuralToy::DataPair dp_2(input_2, target_2);
    neuralToy::DataPair dp_3(input_3, target_3);
    neuralToy::DataPair dp_4(input_4, target_4);

    neuralToy::DataSet ds;
    ds.add(dp_4);
    ds.add(dp_3);
    ds.add(dp_2);
    ds.add(dp_1);

    for (int i = 0; i < 76; ++i) {
        gd.train(ds);
        switch (i) {
        case 0:
            EXPECT_NEAR(0.264738149, gd.calculate_mean_squared_error(ds), 0.00000001);
            break;
        case 1:
            EXPECT_NEAR(0.259062935, gd.calculate_mean_squared_error(ds), 0.00000001);
            break;
        case 2:
            EXPECT_NEAR(0.258331952, gd.calculate_mean_squared_error(ds), 0.00000001);
            break;
        case 75:
            EXPECT_NEAR(0.13860422, gd.calculate_mean_squared_error(ds), 0.00000001);
            break;
        }
    }

    neuralToy::Network n(gd.get_network());
    n.compute(input_1);
    EXPECT_NEAR(0.057976430, n.get_output(1, 0), 0.00000001);
    n.compute(input_2);
    EXPECT_NEAR(0.677635001, n.get_output(1, 0), 0.00000001);
    n.compute(input_3);
    EXPECT_NEAR(0.578385387, n.get_output(1, 0), 0.00000001);
    n.compute(input_4);
    EXPECT_NEAR(0.519015935, n.get_output(1, 0), 0.00000001);

}
