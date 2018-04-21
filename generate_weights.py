import numpy

def generate_weights(input_neurons_number, hidden_neurons_number, output_neurons_number):
    hidden_weights = (numpy.random.random_sample((hidden_neurons_number, input_neurons_number+1)) - 0.0) / 5

    output_weights = (numpy.random.random_sample((output_neurons_number, hidden_neurons_number+1)) - 0.0) / 5

    return hidden_weights, output_weights