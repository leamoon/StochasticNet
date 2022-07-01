#include"bitstring.h"
#include"neuron.h"
#include<iostream>
#include<cmath>
#include<fstream>
using namespace std;

const int bitstring_size = 1024;
double input_layer_array[256];

class identifier
{
public:
    identifier()
    :input_layer_array{},
    input_layer(input_layer_array),
    hidden_layer1(input_layer, "hidden1_weight.txt"),
    hidden_layer2(hidden_layer1, "hidden2_weight.txt"),
    output_layer(hidden_layer2, "out_weight.txt")
    {}
    stochastic_computing_neuron_layer<bitstring_size, 3, 256, 0> input_layer;
    stochastic_computing_neuron_layer<bitstring_size, 3, 32, 256> hidden_layer1;
    stochastic_computing_neuron_layer<bitstring_size, 3, 32, 32> hidden_layer2;
    stochastic_computing_neuron_layer<bitstring_size, 3, 10, 32> output_layer;
    double input_layer_array[256];
    int identify(const char* filename)
    {
        input_array(input_layer_array, filename, 256);
        to_zero_or_one(input_layer_array, 256);
        input_layer.update(input_layer_array);
        hidden_layer1.update(input_layer);
        hidden_layer2.update(hidden_layer1);
        output_layer.update(hidden_layer2);
        return output_layer.max_index();
    }
};

int identify(const char* filename)
{
    input_array(input_layer_array, filename, 256);
    //to_zero_or_one(input_layer_array, 256);
    stochastic_computing_neuron_layer<bitstring_size, 3, 256, 0> input_layer(input_layer_array);
    stochastic_computing_neuron_layer<bitstring_size, 3, 32, 256> hidden_layer1(input_layer, "hidden1_weight.txt");
    stochastic_computing_neuron_layer<bitstring_size, 3, 32, 32> hidden_layer2(hidden_layer1, "hidden2_weight.txt");
    stochastic_computing_neuron_layer<bitstring_size, 3, 10, 32> output_layer(hidden_layer2, "out_weight.txt");
    return output_layer.max_index();
}

int answer[60000];
int main()
{
    ifstream fin("value_list.txt");
    for (int i = 0; i < 100; i++) {
        double x;
        fin >> x;
        answer[i] = static_cast<int>(x);
    }

    identifier id;
    char filename[100];
    char fmt[100] = "data_figures/fig%d.txt";
    int correct_count = 0;
    for (int i = 0; i < 100; i++) {
        sprintf(filename, fmt, i);
        int number = id.identify(filename);
        correct_count += (number == answer[i]);
        cout << number << " " << answer[i] << " " << ((number == answer[i]) ? 'T' : 'F') << "\n";
    }
    cout << correct_count << endl;
    return 0;
}