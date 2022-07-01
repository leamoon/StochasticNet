#pragma once
#include"bitstring.h"
#include<iostream>
#include<cmath>
#include<fstream>
#include<algorithm>

using namespace std;
//ratio class
pair<pair<int, int>,double> to_the_nearest_ratio(double x, int R, int n)
{
    if (x == 0) return {{0, 1}, 0.0};
    if (x == 1) return {{1, 1}, 0.0};
    pair<int, int> result{0, 1};
    double error = INFINITY;
    for (int i = 1; i <= R; i++) {
        if (n <= 1) {
            double a = round(i * x); // a / i ≈ x
            if (a > R) a = R;
            if (a < 0) a = 0;
            double new_error = abs(a / i - x);
            if (new_error < error) {
                result = {a, i};
                error = new_error;
            }
        }
        else {
            auto test = to_the_nearest_ratio(x, R, n - 1);
            if (test.second < error) {
                result = test.first;
                error = test.second;
            }
        }
    }
    return {result, error};
}

template<size_t bitstring_size, size_t M, size_t Size, size_t prev_size>
auto input_weight(const char* filename)
{
    integral_bitstring<M, bitstring_size, bipolar> (*Result)[prev_size] = new integral_bitstring<M, bitstring_size, bipolar>[Size][prev_size];
    ifstream fin(filename);
    for (int i = 0; i < Size; i++) {
        for (int j = 0; j < prev_size; j++) {
            double x;
            fin >> x;
            Result[i][j] = integral_bitstring<M, bitstring_size, bipolar>(x);
        }
    }
    fin.close();
    return Result;
}

template<size_t bitstring_size, size_t M, size_t Size, size_t prev_size>
auto input_weight_precision(const char* filename, double precision, double factor = 1.0)
{
    integral_bitstring<M, bitstring_size, bipolar> (*Result)[prev_size] = new integral_bitstring<M, bitstring_size, bipolar>[Size][prev_size];
    ifstream fin(filename);
    for (int i = 0; i < Size; i++) {
        for (int j = 0; j < prev_size; j++) {
            double x;
            fin >> x;
            x = round(x / precision) * precision * factor;
            Result[i][j] = integral_bitstring<M, bitstring_size, bipolar>(x);
        }
    }
    fin.close();
    return Result;
}

pair<int,int> nearest_fractor(double x, int N)
{
    double err = INFINITY;
    int a = 0, b = 0;
    for (int j = 1; j <= N; j++) {
        int i = round(x * j);
        double current_err = abs(i * 1.0 / j - x);
        if (current_err < err) {
            a = i;
            b = j;
            err = current_err;
        }
    }
    return make_pair(a, b);
}

template<size_t bitstring_size, size_t M, size_t Size, size_t prev_size>
auto input_weight_fraction(const char* filename, int denominator_N, double factor = 1.0)
{
    integral_bitstring<M, bitstring_size, bipolar> (*Result)[prev_size] = new integral_bitstring<M, bitstring_size, bipolar>[Size][prev_size];
    ifstream fin(filename);
    for (int i = 0; i < Size; i++) {
        for (int j = 0; j < prev_size; j++) {
            double x;
            fin >> x;
            auto nearest_fractor_frequency = nearest_fractor((x * factor / M + 1) / 2, denominator_N);
            x = nearest_fractor_frequency.first * 1.0 / nearest_fractor_frequency.second * 2 - 1;
            x = x * M;
            Result[i][j] = integral_bitstring<M, bitstring_size, bipolar>(x);
        }
    }
    fin.close();
    return Result;
}

void input_array(double input[], const char* filename, int N)
{
    ifstream fin(filename);
    for (int i = 0; i < N; i++) fin >> input[i];
    fin.close();
}

void input_array(double input[], const char* filename, int N, double precision, double factor)
{
    ifstream fin(filename);
    for (int i = 0; i < N; i++) {
        double x;
        fin >> x;
        x = round(x / precision) * precision * factor;
        input[i] = x;
    }
    fin.close();
}

template<size_t M>
void input_array_2d(double input[][M], const char* filename, int N, double precision = 0.01, double factor = 1.0)
{
    ifstream fin(filename);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            double x;
            fin >> x;
            x = round(x / precision) * precision * factor;
            // if (x > 1) x = 1;
            input[i][j] = x;
        }
    }
    fin.close();
}

void to_zero_or_one(double input[], int N)
{
    for (int i = 0; i < N; i++) {
        if (input[i] > 0.5) input[i] = 1;
        else input[i] = 0;
    }
}

template<int M>
void to_zero_or_one_2d(double input[][M], int N, double cond=0.5)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (input[i][j] > cond) input[i][j] = 1.0;
            else input[i][j] = 0.0;
        }
    }
}

template<size_t bitstring_size, size_t M, size_t Size, size_t prev_size>
class stochastic_computing_neuron_layer
{
public:
    template<size_t prev_prev_size, size_t prev_M>
    stochastic_computing_neuron_layer(const stochastic_computing_neuron_layer<bitstring_size,prev_M,prev_size,prev_prev_size>& previous_layer, const char* filename)
    :LinearTrans(input_weight<bitstring_size,M,Size,prev_size>(filename))
    {
        for (int i = 0; i < Size; i++) {
            integral_bitstring<M * prev_size, bitstring_size, bipolar> output = sum_of_array_with_weigth<M, prev_size, bitstring_size, bipolar>(LinearTrans[i], previous_layer.Output);
            Output[i] = bitstring<bitstring_size, bipolar>(output.NStanh(2));
        }
        previous_size = prev_size;
    }
    template<size_t prev_prev_size, size_t prev_M>
    stochastic_computing_neuron_layer(const stochastic_computing_neuron_layer<bitstring_size,prev_M,prev_size,prev_prev_size>& previous_layer, const char* filename, int tanh_N, double precision, double factor)
    :LinearTrans(input_weight_precision<bitstring_size,M,Size,prev_size>(filename, precision, factor))
    {
        for (int i = 0; i < Size; i++) {
            integral_bitstring<M * prev_size, bitstring_size, bipolar> output = sum_of_array_with_weigth<M, prev_size, bitstring_size, bipolar>(LinearTrans[i], previous_layer.Output);
            Output[i] = bitstring<bitstring_size, bipolar>(output.NStanh(tanh_N));
        }
        previous_size = prev_size;
    }
    template<size_t prev_prev_size, size_t prev_M>
    stochastic_computing_neuron_layer(const stochastic_computing_neuron_layer<bitstring_size,prev_M,prev_size,prev_prev_size>& previous_layer, const char* filename, int tanh_N, int denominator_N, double factor, int bound)
    :LinearTrans(input_weight_fraction<bitstring_size,M,Size,prev_size>(filename, denominator_N, factor))
    {
        for (int i = 0; i < Size; i++) {
            integral_bitstring<M * prev_size, bitstring_size, bipolar> output = sum_of_array_with_weigth<M, prev_size, bitstring_size, bipolar>(LinearTrans[i], previous_layer.Output);
            Output[i] = bitstring<bitstring_size, bipolar>(output.NStanh_bound(tanh_N, bound));
        }
        previous_size = prev_size;
    }
    stochastic_computing_neuron_layer(double const_output[])
    {
        for (int i = 0; i < Size; i++) Output[i] = bitstring<bitstring_size, bipolar>(const_output[i]);
    }
    ~stochastic_computing_neuron_layer() { delete LinearTrans; }
    void output_value() const
    {
        for (int i = 0; i < Size; i++) cout << Output[i].value() << " ";
        cout << "\n";
    }
    void output_weight() const
    {
        cout << "Wei\n";
        for (int i = 0; i < Size; i++) {
            for (int j = 0; j < previous_size; j++) cout << LinearTrans[j][i].value() << " ";
            cout << endl;
        }
    }
    int max_index() const
    {
        return max_element(begin(Output), end(Output), 
            [](const bitstring<bitstring_size, bipolar>& a, const bitstring<bitstring_size, bipolar>& b)
            {
                return a.value() < b.value();
            }
        ) - begin(Output);
    }
    template<size_t prev_prev_size, size_t prev_M>
    void update(const stochastic_computing_neuron_layer<bitstring_size,prev_M,prev_size,prev_prev_size>& previous_layer)
    {
        for (int i = 0; i < Size; i++) {
            integral_bitstring<M * prev_size, bitstring_size, bipolar> output = sum_of_array_with_weigth<M, prev_size, bitstring_size, bipolar>(LinearTrans[i], previous_layer.Output);
            Output[i] = bitstring<bitstring_size, bipolar>(output.NStanh(2));
        }
    }
    void update(double const_output[])
    {
        for (int i = 0; i < Size; i++) Output[i] = bitstring<bitstring_size, bipolar>(const_output[i]);
    }
    int previous_size;
    integral_bitstring<M, bitstring_size, bipolar> (*LinearTrans)[prev_size]; // LinearTrans[previous_size][self_size]
    bitstring<bitstring_size, bipolar> Output[Size];
};
