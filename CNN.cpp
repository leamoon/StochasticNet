#include"bitstring.h"
#include"neuron.h"
#include<iostream>
#include<cmath>
#include<fstream>
using namespace std;

const int bitstring_size = 256;
double convolution_result[49];
double input[15][15];
double kernel[3][3];

class identifier_CNN
{
public:
    identifier_CNN()
    :convolution_result{},
    input_layer(convolution_result),
    hidden_layer1(input_layer, "CNN weights/fc1_weight.txt", 2, 10, 1, 128),
    hidden_layer2(hidden_layer1, "CNN weights/fc2_weight.txt", 2, 10, 1, 64),
    output_layer(hidden_layer2, "CNN weights/fc3_weight.txt", 2, 10, 1, 64)
    {
        input_array_2d(kernel, "CNN weights/conv1_weight.txt", 3, 1.0/8, 0.5);
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) cout << kernel[i][j]*16 << " "; cout << endl;
        // for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) if(kernel[i][j] < 0) kernel[i][j] = 0;
    }
    stochastic_computing_neuron_layer<bitstring_size, 2, 49, 0> input_layer;
    stochastic_computing_neuron_layer<bitstring_size, 1, 30, 49> hidden_layer1;
    stochastic_computing_neuron_layer<bitstring_size, 1, 30, 30> hidden_layer2;
    stochastic_computing_neuron_layer<bitstring_size, 1, 10, 30> output_layer;
    double input_image[16][16];
    double kernel[3][3];
    double convolution_result[49];
    int identify(const char* filename)
    {
        input_array_2d(input_image, filename, 16);
        to_zero_or_one_2d(input_image, 16, 0.05);
        conv_and_relu();
        input_layer.update(convolution_result);
        hidden_layer1.update(input_layer);
        hidden_layer2.update(hidden_layer1);
        output_layer.update(hidden_layer2);
        //output_layer.output_value();
        return output_layer.max_index();
    }
    void conv_and_relu()
    {
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                int cij = i * 7 + j, i2 = i<<1, j2 = j<<1;
                convolution_result[cij] = 0;
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        convolution_result[cij] += kernel[k][l] * input_image[i2 + k][j2 + l];
                    }
                }
                if (convolution_result[cij] < 0) convolution_result[cij] = 0;
                if (convolution_result[cij] > 1) convolution_result[cij] = 1;
                //cout << convolution_result[cij] * 16 << " ";
            }
        //cout << endl;
        }
    }
};

int identify(const char* filename)
{
    //to_zero_or_one(input_layer_array, 256);
    stochastic_computing_neuron_layer<bitstring_size, 3, 49, 0> input_layer(convolution_result);
    stochastic_computing_neuron_layer<bitstring_size, 3, 30, 49> hidden_layer1(input_layer, "fc1_weight.txt");
    stochastic_computing_neuron_layer<bitstring_size, 3, 30, 30> hidden_layer2(hidden_layer1, "fc2_weight.txt");
    stochastic_computing_neuron_layer<bitstring_size, 3, 10, 30> output_layer(hidden_layer2, "fc3_weight.txt");
    return output_layer.max_index();
}

int answer[60000];
int main()
{
    ifstream fin("value_list.txt");
    for (int i = 0; i < 10000; i++) {
        double x;
        fin >> x;
        answer[i] = static_cast<int>(x);
    }

    identifier_CNN id;
    char filename[100];
    char fmt[100] = "data_figures/fig%d.txt";
    int correct_count = 0;
    for (int i = 0; i < 2000; i++) {
        sprintf(filename, fmt, i);
        int number = id.identify(filename);
        //int number = id.identify("test_written_number.txt");
        correct_count += (number == answer[i]);
        cout << number << " " << answer[i] << " " << ((number == answer[i]) ? 'T' : 'F') << "\n";
    }
    cout << correct_count << endl;
    // cout << id.identify("data_figures/fig0.txt") << endl;
    // id.output_layer.output_value();
    return 0;
}