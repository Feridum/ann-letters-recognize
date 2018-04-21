import numpy

beta= 1
hidden_bias = 0
output_bias = 0


def recognize(input_neurons, hidden_neurons_number, output_neurons_number, hidden_weights, output_weights):
    hidden_neurons = numpy.full(hidden_neurons_number, 0.0)
    output_neurons = numpy.full(output_neurons_number, 0.0)

    for x in range(0, hidden_neurons_number):
        hidden_neurons[x] = (
                    1.0 / (1.0 + numpy.exp(-beta * (numpy.sum(input_neurons * hidden_weights[x]) + hidden_bias))))

    hidden_neurons = numpy.insert(hidden_neurons, 0, 1)
    for y in range(0, output_neurons_number):
        output_neurons[y] = (
                    1.0 / (1.0 + numpy.exp(-beta * (numpy.sum(hidden_neurons * output_weights[y]) + output_bias))))

    return hidden_neurons, output_neurons