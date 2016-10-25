#include <gtest/gtest.h>

#include "resilientpropagation.h"

TEST(ResilientPropagationTest, train)
{
    std::queue<unsigned short> hidden_layers_size;
    hidden_layers_size.push(2);
    neuralToy::Network network(2, hidden_layers_size, 1);

    network.adjust_layer(0).set_transferFunction(
                                        neuralToy::TransferFunction::LOGISTIC);
    network.adjust_layer(1).set_transferFunction(
                                        neuralToy::TransferFunction::LOGISTIC);

    network.adjust_layer(0).adjust_neuron(0).set_weight(0, 0.15);
    network.adjust_layer(0).adjust_neuron(0).set_weight(1, 0.20);
    network.adjust_layer(0).adjust_neuron(0).set_weight(2, 0.35);

    network.adjust_layer(0).adjust_neuron(1).set_weight(0, 0.25);
    network.adjust_layer(0).adjust_neuron(1).set_weight(1, 0.30);
    network.adjust_layer(0).adjust_neuron(1).set_weight(2, 0.35);

    network.adjust_layer(1).adjust_neuron(0).set_weight(0, 0.40);
    network.adjust_layer(1).adjust_neuron(0).set_weight(1, 0.45);
    network.adjust_layer(1).adjust_neuron(0).set_weight(2, 0.50);

    neuralToy::ResilientPropagation rp(network);

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
        rp.train(ds);
        switch (i) {
        case 0:
            /*
            EXPECT_NEAR(-0.0125, rp.branch_[0][0][0].dw, 0.00000001);
            EXPECT_NEAR(-0.0125, rp.branch_[0][0][1].dw, 0.00000001);
            EXPECT_NEAR(-0.0125, rp.branch_[0][0][2].dw, 0.00000001);

            EXPECT_NEAR(-0.0125, rp.branch_[0][1][0].dw, 0.00000001);
            EXPECT_NEAR(-0.0125, rp.branch_[0][1][1].dw, 0.00000001);
            EXPECT_NEAR(-0.0125, rp.branch_[0][1][2].dw, 0.00000001);

            EXPECT_NEAR(-0.0125, rp.branch_[1][0][0].dw, 0.00000001);
            EXPECT_NEAR(-0.0125, rp.branch_[1][0][1].dw, 0.00000001);
            EXPECT_NEAR(-0.0125, rp.branch_[1][0][2].dw, 0.00000001);

            EXPECT_NEAR(0.1375, rp.network_->get_layer(0).get_neuron(0).get_weight(0), 0.00000001);
            EXPECT_NEAR(0.1875, rp.network_->get_layer(0).get_neuron(0).get_weight(1), 0.00000001);
            EXPECT_NEAR(0.3375, rp.network_->get_layer(0).get_neuron(0).get_weight(2), 0.00000001);

            EXPECT_NEAR(0.2375, rp.network_->get_layer(0).get_neuron(1).get_weight(0), 0.00000001);
            EXPECT_NEAR(0.2875, rp.network_->get_layer(0).get_neuron(1).get_weight(1), 0.00000001);
            EXPECT_NEAR(0.3375, rp.network_->get_layer(0).get_neuron(1).get_weight(2), 0.00000001);

            EXPECT_NEAR(0.3875, rp.network_->get_layer(1).get_neuron(0).get_weight(0), 0.00000001);
            EXPECT_NEAR(0.4375, rp.network_->get_layer(1).get_neuron(0).get_weight(1), 0.00000001);
            EXPECT_NEAR(0.4875, rp.network_->get_layer(1).get_neuron(0).get_weight(2), 0.00000001);
            */
            EXPECT_NEAR(0.304198731, rp.calculate_mean_squared_error(ds), 0.00000001);
            break;
        case 1:
            EXPECT_NEAR(0.300656696, rp.calculate_mean_squared_error(ds), 0.00000001);
            break;
        case 2:
            EXPECT_NEAR(0.296523355, rp.calculate_mean_squared_error(ds), 0.00000001);
            break;
        case 75:
            EXPECT_NEAR(0.079352499, rp.calculate_mean_squared_error(ds), 0.00000001);
            break;
        }
    }
}
