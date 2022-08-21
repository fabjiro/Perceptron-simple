#include <vector>
#include <random>
#include <math.h>

#pragma once
class Perceptron
{	
private:
	float umbral;
	std::vector<float> weights;
	float acivation(const float&)noexcept;
	float random() noexcept;
public:
	explicit Perceptron(const int32_t&, const float&) noexcept;
	float predict(const std::vector<float>&) noexcept;
	void fit(const int32_t&, const std::vector<std::vector<float>>&, const std::vector<float>&) noexcept;
};


inline float Perceptron::acivation(const float& val) noexcept
{
	return std::tanh(-val);
}

Perceptron::Perceptron(const int32_t & n_neuro, const float & eta = 0.1) noexcept
{
	this->umbral = eta;
	for (int32_t i = 0; i < n_neuro; ++i) {
		weights.push_back(this->random() - random());
	}
}

float Perceptron::predict(const std::vector<float>& inputs) noexcept {
	float sum = this->umbral;
	for (int32_t i = 0; i < this->weights.size(); i++) {
		sum += inputs.at(i) * this->weights.at(i);
	}
	return this->acivation(sum);
}

void Perceptron::fit(
	const int32_t & epochs, 
	const std::vector<std::vector<float>>& inputs, 
	const std::vector<float>& outputs) noexcept
{
	float prediction{}, error{};

	for (int32_t _ = 0; _ < epochs; ++_) {
		for (int32_t i = 0; i < inputs.size(); i++) {
			prediction = this->predict(inputs.at(i));
			error = prediction - outputs.at(i);

			for (int32_t j = 0; j < this->weights.size(); j++) {
				this->weights[j] += this->umbral * error * inputs[i][j];
			}
		}
	}
}

float Perceptron::random() noexcept {
	return ((float)rand()) / ((float)RAND_MAX);
}