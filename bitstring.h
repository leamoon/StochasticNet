#ifndef BITSTRING_H
#define BITSTRING_H
#include<bitset>
#include<random>
#include<chrono>
#include<array>
#include<iostream>
using namespace std;

enum bitstring_type {unipolar, bipolar};

default_random_engine random_generator(chrono::high_resolution_clock::now().time_since_epoch().count());

template<size_t bitstring_size, bitstring_type T>
class bitstring
{
public:
    bitset<bitstring_size> X;
    bitstring() {}
    bitstring(double x);
    bitstring(bitset<bitstring_size>&& S): X(S) {}
    ~bitstring();
    double value() const;
    bitstring<bitstring_size, T> operator|(bitstring<bitstring_size, T> y) { return bitstring<bitstring_size, T>(X | y.X);}
    void operator|=(bitstring<bitstring_size, T> y) { X |= y.X;}
    bitstring<bitstring_size, T> operator&(bitstring<bitstring_size, T> y) { return bitstring<bitstring_size, T>(X & y.X);}
    void operator&=(bitstring<bitstring_size, T> y) { X &= y.X;}
    bitstring<bitstring_size, T> operator^(bitstring<bitstring_size, T> y) { return bitstring<bitstring_size, T>(X ^ y.X);}
    void operator^=(bitstring<bitstring_size, T> y) { X ^= y.X;}
    bitstring<bitstring_size, unipolar> operator*(const bitstring<bitstring_size, unipolar>& y)
    {
        return bitstring<bitstring_size, unipolar>(X & y.X);
    }
    bitstring<bitstring_size, bipolar> operator*(const bitstring<bitstring_size, bipolar>& y)
    {
        return bitstring<bitstring_size, bipolar>(~(X ^ y.X));
    }
    bitstring<bitstring_size, T> Stanh(int n);
    bool at(size_t position) { return X.test(position); }
};

template<size_t bitstring_size, bitstring_type T>
bitstring<bitstring_size, T>::bitstring(double x)
{
    if (T == unipolar) {
        bernoulli_distribution R(x);
        for (size_t i = 0; i < bitstring_size; i++) X.set(i, R(random_generator));
    }
    else {
        bernoulli_distribution R((x + 1) / 2.0);
        for (size_t i = 0; i < bitstring_size; i++) X.set(i, R(random_generator));
    }
}

template<size_t bitstring_size, bitstring_type T>
bitstring<bitstring_size, T>::~bitstring()
{
}

template<size_t bitstring_size, bitstring_type T>
double bitstring<bitstring_size, T>::value() const
{
    double unipolar_value = static_cast<double>(X.count()) / bitstring_size;
    if (T == unipolar) return unipolar_value;
    return 2 * unipolar_value - 1;
}

template<size_t bitstring_size, bitstring_type T>
bitstring<bitstring_size, T> bitstring<bitstring_size, T>::Stanh(int n)
{
    bitset<bitstring_size> Y;
    int count = n / 2 - 1;
    for (int i = 0; i < bitstring_size; i++) {
        count += 2 * X.test(i) - 1;
        if (count > n - 1) count = n - 1;
        if (count < 0) count = 0;
        if (count > n / 2 - 1) Y.set(i, 1);
    }
    return bitstring<bitstring_size, T>(move(Y));
}

template<size_t M, size_t bitstring_size, bitstring_type T>
class integral_bitstring
{
public:
    array<bitstring<bitstring_size, T>,M> X;
    integral_bitstring() {};
    integral_bitstring(double x)
    {
        // X.fill(bitstring<bitstring_size, T>(x / M));
        for (int i = 0; i < M; i++) X[i] = bitstring<bitstring_size, T>(x / M);
    }
    ~integral_bitstring() {};
    double value() const
    {
        double v = 0;
        for (auto& x: X) v += x.value();
        return v;
    }
    size_t size() const { return M; }
    template<size_t M2>
    integral_bitstring<M+M2, bitstring_size, T> operator+(integral_bitstring<M2, bitstring_size, T> y)
    {
        integral_bitstring<M+M2, bitstring_size, T> Z;
        for (int i = 0; i < M; i++) Z.X[i] = X[i];
        for (int i = M; i < M + M2; i++) Z.X[i] = y.X[i - M];
        return move(Z);
    }
    integral_bitstring<M, bitstring_size, T> operator*(const bitstring<bitstring_size, T>& y)
    {
        integral_bitstring<M, bitstring_size, T> Z;
        for (int i = 0; i < M; i++) Z.X[i] = X[i] * y;
        return move(Z);
    }
    template<size_t M2>
    integral_bitstring<M*M2, bitstring_size, T> operator*(integral_bitstring<M2, bitstring_size, T> y)
    {
        integral_bitstring<M*M2, bitstring_size, T> Z;
        for (int i = 0; i < M; i++) 
            for (int j = 0; j < M2; j++) Z.X[i * M2 + j] = X[i] * y.X[j];
        return move(Z);
    }
    int at(size_t position) const
    {
        int s = 0;
        for (auto &x: X) s += x.X.test(position);
        return s;
    }
    bitstring<bitstring_size, T> NStanh(int n) const;
    bitstring<bitstring_size, T> NStanh_bound(int n, int bound) const;
};

template<size_t M, size_t bitstring_size, bitstring_type T>
bitstring<bitstring_size, T> integral_bitstring<M, bitstring_size, T>::NStanh(int n) const
{
    bitset<bitstring_size> Y;
    int count = n * M / 2 - 1;
    for (int i = 0; i < bitstring_size; i++) {
        count += 2 * at(i) - static_cast<int>(M);
        if (count > static_cast<int>(n * M - 1)) count = n * M - 1;
        if (count < 0) count = 0;
        if (count > static_cast<int>(n * M / 2 - 1)) Y.set(i, 1);
    }
    return bitstring<bitstring_size, T>(move(Y));
}

template<size_t M, size_t bitstring_size, bitstring_type T>
bitstring<bitstring_size, T> integral_bitstring<M, bitstring_size, T>::NStanh_bound(int n, int bound) const
{
    bitset<bitstring_size> Y;
    int count = bound / 2 - 1;
    for (int i = 0; i < bitstring_size; i++) {
        count += 2 * at(i) - static_cast<int>(M);
        if (count > static_cast<int>(bound - 1)) count = bound - 1;
        if (count < 0) count = 0;
        if (count > static_cast<int>(bound / 2 - 1)) Y.set(i, 1);
    }
    return bitstring<bitstring_size, T>(move(Y));
}

template<size_t M, size_t N, size_t bitstring_size, bitstring_type T>
integral_bitstring<M*N, bitstring_size, T> sum_of_array(integral_bitstring<M, bitstring_size, T> arr[])
{
    integral_bitstring<M*N, bitstring_size, T> result;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            result.X[i * M + j] = arr[i].X[j];
        }
    }
    return move(result);
}

template<size_t M, size_t N, size_t bitstring_size, bitstring_type T>
integral_bitstring<M*N, bitstring_size, T> sum_of_array_with_weigth(integral_bitstring<M, bitstring_size, T> arr[N], const bitstring<bitstring_size, T> w[N])
{
    integral_bitstring<M*N, bitstring_size, T> result;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            result.X[i * M + j] = arr[i].X[j] * w[i];
        }
    }
    return move(result);
}

template<size_t M, size_t N, size_t bitstring_size, bitstring_type T>
integral_bitstring<M*N, bitstring_size, T> test_func(const bitstring<bitstring_size, T> w[N])
{
    integral_bitstring<M*N, bitstring_size, T> result;
    return move(result);
}
#endif