#include <cstdio>
#include <iostream>
#include <queue>

#include "gradientDescent.h"
#include "resilientpropagation.h"

std::default_random_engine generator_;

int main()
{
    std::queue<unsigned short> hidden_layers_size;
    hidden_layers_size.push(2);
    neuralToy::Network network(2, hidden_layers_size, 1);

    network.adjust_layer(0).set_transferFunction(
                                        neuralToy::TransferFunction::LOGISTIC);
    network.adjust_layer(1).set_transferFunction(
                                        neuralToy::TransferFunction::LOGISTIC);

    network.reset_weights();

    // XOR
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

    neuralToy::ResilientPropagation rprop(network, 0.1);

    neuralToy::GradientDescent gd_online(network, neuralToy::Learning::ONLINE);
    gd_online.set_learning_rate(0.5);
    gd_online.set_momentum(0.99);

    neuralToy::GradientDescent gd_batch(network, neuralToy::Learning::BATCH);
    gd_batch.set_learning_rate(0.6);
    gd_batch.set_momentum(0.99);

    for (int epoch = 0; epoch < 200; ++epoch) {
        rprop.train(ds);
        gd_online.train(ds);
        gd_batch.train(ds);

        std::printf("epoch #%i\
                    \n\tmse for iRPROP+:\t\t\t\t%e\
                    mse for GRADIENT DESCENT (online-learning): \t%e\
                    mse for GRADIENT DESCENT (batch-learning): \t%e\n",
                    epoch,
                    rprop.calculate_mean_squared_error(ds),
                    gd_online.calculate_mean_squared_error(ds),
                    gd_batch.calculate_mean_squared_error(ds));
    }

    /*
    cout << "input: " << input_1[0] << " " << input_1[1] << " output=" << network.compute(input_1)[0] << endl;
    cout << "input: " << input_2[0] << " " << input_2[1] << " output=" << network.compute(input_2)[0] << endl;
    cout << "input: " << input_3[0] << " " << input_3[1] << " output=" << network.compute(input_3)[0] << endl;
    cout << "input: " << input_4[0] << " " << input_4[1] << " output=" << network.compute(input_4)[0] << endl;
    */

    return 0;
}
