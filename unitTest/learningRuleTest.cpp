#include <gtest/gtest.h>

#include "learningRule.h"

namespace neuralToy
{
    class LearningRuleDerived : public LearningRule
    {
    public:
        LearningRuleDerived(const Network& network, Learning learning)
            : LearningRule(network, learning)
        {
        }

        ~LearningRuleDerived()
        {
        }

        void train(const DataSet& ds)
        {
        }

        void backward_pass(const DataPair& dataPair)
        {
            calculate_backward_pass(dataPair);
        }

        const double& error(unsigned short layer_index,
                            unsigned short neuron_index) const
        {
            return get_error(layer_index, neuron_index);
        }

        const double& gradient(unsigned short layer_index,
                               unsigned short neuron_index,
                               unsigned short weight_index) const
        {
            return get_gradient(layer_index, neuron_index, weight_index);
        }
    };
}

TEST(LearningRuleTest, calculate_derivative)
{
    EXPECT_NEAR(-3.9525,
                neuralToy::LearningRule::calculate_derivative(
                    -1.55,
                    neuralToy::TransferFunction::LOGISTIC),
                0.0001);
    EXPECT_NEAR(-1.4025,
                neuralToy::LearningRule::calculate_derivative(
                    -1.55,
                    neuralToy::TransferFunction::HYPERBOLIC_TANGENT),
                0.000000001);
    EXPECT_EQ(1, neuralToy::LearningRule::calculate_derivative(-1.55,
              neuralToy::TransferFunction::LINEAR));
}

TEST(LearningRuleTest, calculate_mean_squared_error)
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

    neuralToy::LearningRuleDerived lrd(network, neuralToy::Learning::ONLINE);

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
    ds.add(dp_1);
    ds.add(dp_2);
    ds.add(dp_3);
    ds.add(dp_4);

    EXPECT_NEAR(0.30721275, lrd.calculate_mean_squared_error(ds), 0.00000001);
}

TEST(LearningRuleTest, get_error)
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

    neuralToy::LearningRuleDerived lrd(network, neuralToy::Learning::ONLINE);

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

    lrd.backward_pass(dp_1);
    EXPECT_NEAR(-0.14377295, -lrd.error(1, 0), 0.00000001);
    EXPECT_NEAR(-0.01394583, -lrd.error(0, 0), 0.00000001);
    EXPECT_NEAR(-0.01568905, -lrd.error(0, 1), 0.00000001);

    lrd.backward_pass(dp_2);
    EXPECT_NEAR(0.04982039, -lrd.error(1, 0), 0.00000001);
    EXPECT_NEAR(0.00462348, -lrd.error(0, 0), 0.00000001);
    EXPECT_NEAR(0.00505211, -lrd.error(0, 1), 0.00000001);

    lrd.backward_pass(dp_3);
    EXPECT_NEAR(0.05041794, -lrd.error(1, 0), 0.00000001);
    EXPECT_NEAR(0.00473936, -lrd.error(0, 0), 0.00000001);
    EXPECT_NEAR(0.00519067, -lrd.error(0, 1), 0.00000001);

    lrd.backward_pass(dp_4);
    EXPECT_NEAR(-0.14102153, -lrd.error(1, 0), 0.00000001);
    EXPECT_NEAR(-0.01250652, -lrd.error(0, 0), 0.00000001);
    EXPECT_NEAR(-0.01304099, -lrd.error(0, 1), 0.00000001);
}

TEST(LearningRuleTest, get_gradient)
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

    neuralToy::LearningRuleDerived lrd(network, neuralToy::Learning::ONLINE);

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

    double gr[2][2][3]{0};
    lrd.backward_pass(dp_1);
    gr[0][0][0] += lrd.gradient(0, 0, 0);
    gr[0][0][1] += lrd.gradient(0, 0, 1);
    gr[0][0][2] += lrd.gradient(0, 0, 2);

    gr[0][1][0] += lrd.gradient(0, 1, 0);
    gr[0][1][1] += lrd.gradient(0, 1, 1);
    gr[0][1][2] += lrd.gradient(0, 1, 2);

    gr[1][0][0] += lrd.gradient(1, 0, 0);
    gr[1][0][1] += lrd.gradient(1, 0, 1);
    gr[1][0][2] += lrd.gradient(1, 0, 2);
    lrd.backward_pass(dp_2);
    gr[0][0][0] += lrd.gradient(0, 0, 0);
    gr[0][0][1] += lrd.gradient(0, 0, 1);
    gr[0][0][2] += lrd.gradient(0, 0, 2);

    gr[0][1][0] += lrd.gradient(0, 1, 0);
    gr[0][1][1] += lrd.gradient(0, 1, 1);
    gr[0][1][2] += lrd.gradient(0, 1, 2);

    gr[1][0][0] += lrd.gradient(1, 0, 0);
    gr[1][0][1] += lrd.gradient(1, 0, 1);
    gr[1][0][2] += lrd.gradient(1, 0, 2);
    lrd.backward_pass(dp_3);
    gr[0][0][0] += lrd.gradient(0, 0, 0);
    gr[0][0][1] += lrd.gradient(0, 0, 1);
    gr[0][0][2] += lrd.gradient(0, 0, 2);

    gr[0][1][0] += lrd.gradient(0, 1, 0);
    gr[0][1][1] += lrd.gradient(0, 1, 1);
    gr[0][1][2] += lrd.gradient(0, 1, 2);

    gr[1][0][0] += lrd.gradient(1, 0, 0);
    gr[1][0][1] += lrd.gradient(1, 0, 1);
    gr[1][0][2] += lrd.gradient(1, 0, 2);
    lrd.backward_pass(dp_4);
    gr[0][0][0] += lrd.gradient(0, 0, 0);
    gr[0][0][1] += lrd.gradient(0, 0, 1);
    gr[0][0][2] += lrd.gradient(0, 0, 2);

    gr[0][1][0] += lrd.gradient(0, 1, 0);
    gr[0][1][1] += lrd.gradient(0, 1, 1);
    gr[0][1][2] += lrd.gradient(0, 1, 2);

    gr[1][0][0] += lrd.gradient(1, 0, 0);
    gr[1][0][1] += lrd.gradient(1, 0, 1);
    gr[1][0][2] += lrd.gradient(1, 0, 2);

    EXPECT_NEAR(-0.00776715, -gr[0][0][0], 0.00000001);
    EXPECT_NEAR(-0.00788303, -gr[0][0][1], 0.00000001);
    EXPECT_NEAR(-0.01708950, -gr[0][0][2], 0.00000001);

    EXPECT_NEAR(-0.00785031, -gr[0][1][0], 0.00000001);
    EXPECT_NEAR(-0.00798888, -gr[0][1][1], 0.00000001);
    EXPECT_NEAR(-0.01848726, -gr[0][1][2], 0.00000001);

    EXPECT_NEAR(-0.11559260, -gr[1][0][0], 0.00000001);
    EXPECT_NEAR(-0.11931374, -gr[1][0][1], 0.00000001);
    EXPECT_NEAR(-0.18455614, -gr[1][0][2], 0.00000001);
}

TEST(LearningRuleTest, calculate_backward_pass)
{
    std::queue<unsigned short> hidden_layers_size;
    hidden_layers_size.push(2);
    neuralToy::Network network(2, hidden_layers_size, 2);

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
    network.adjust_layer(1).adjust_neuron(0).set_weight(2, 0.6);

    network.adjust_layer(1).adjust_neuron(1).set_weight(0, 0.5);
    network.adjust_layer(1).adjust_neuron(1).set_weight(1, 0.55);
    network.adjust_layer(1).adjust_neuron(1).set_weight(2, 0.6);

    neuralToy::LearningRuleDerived lrd(network, neuralToy::Learning::ONLINE);

    neuralToy::Pattern input(2);
    input[0] = 0.05;
    input[1] = 0.1;
    neuralToy::Pattern target(2);
    target[0] = 0.01;
    target[1] = 0.99;
    neuralToy::DataPair dp(input, target);
    lrd.backward_pass(dp);

    EXPECT_NEAR(0.008771354, lrd.error(0, 0), 0.00000001);
    EXPECT_NEAR(0.009954254, lrd.error(0, 1), 0.00000001);

    EXPECT_NEAR(0.138498562, lrd.error(1, 0), 0.00000001);
    EXPECT_NEAR(-0.038098236, lrd.error(1, 1), 0.00000001);

    EXPECT_NEAR(0.000438567, lrd.gradient(0, 0, 0), 0.00000001);
    EXPECT_NEAR(0.000877135, lrd.gradient(0, 0, 1), 0.00000001);
    EXPECT_NEAR(0.008771354, lrd.gradient(0, 0, 2), 0.00000001); // bias

    EXPECT_NEAR(0.000497712, lrd.gradient(0, 1, 0), 0.00000001);
    EXPECT_NEAR(0.000995425, lrd.gradient(0, 1, 1), 0.00000001);
    EXPECT_NEAR(0.009954254, lrd.gradient(0, 1, 2), 0.00000001); // bias

    EXPECT_NEAR(0.082167040, lrd.gradient(1, 0, 0), 0.00000001);
    EXPECT_NEAR(0.082667627, lrd.gradient(1, 0, 1), 0.00000001);
    EXPECT_NEAR(0.138498561, lrd.gradient(1, 0, 2), 0.00000001); // bias

    EXPECT_NEAR(-0.022602540, lrd.gradient(1, 1, 0), 0.00000001);
    EXPECT_NEAR(-0.022740242, lrd.gradient(1, 1, 1), 0.00000001);
    EXPECT_NEAR(-0.038098236, lrd.gradient(1, 1, 2), 0.00000001); // bias
}

