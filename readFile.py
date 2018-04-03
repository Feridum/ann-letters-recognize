from scipy import misc
import numpy

# input_neurons_number = 16384
input_neurons_number = 49
hidden_neurons_number = 10
# hidden_neurons_number = 200
output_neurons_number = 2
beta = 1  # .01-5
eta = 0.3  # 0.01 - 0.6
alfa = 0.5  # 0-1


def recognize(path, hidden_weights, output_weights):
    hidden_neurons = numpy.full(hidden_neurons_number, 0.0)
    output_neurons = numpy.full(output_neurons_number, 0.0)

    image = misc.imread(path, True, "L")
    input_neurons = numpy.array(image.flatten() / 255)

    for x in range(0, hidden_neurons_number):
        hidden_neurons[x] = (
                    1.0 / (1.0 + numpy.exp(-beta * (numpy.sum(input_neurons * hidden_weights[x]) + hidden_bias))))

    for y in range(0, output_neurons_number):
        output_neurons[y] = (
                    1.0 / (1.0 + numpy.exp(-beta * (numpy.sum(hidden_neurons * output_weights[y]) + output_bias))))

    return input_neurons, hidden_neurons, output_neurons


numpy.set_printoptions(threshold=numpy.nan)

hidden_weights = (numpy.random.random_sample((hidden_neurons_number, input_neurons_number)) - 0.0) / 5

output_weights = (numpy.random.random_sample((output_neurons_number, hidden_neurons_number)) - 0.0) / 5

e_output = numpy.full((output_neurons_number, hidden_neurons_number), 0.0)
e_hidden = numpy.full((hidden_neurons_number, input_neurons_number), 0.0)

b = numpy.full((output_neurons_number, output_neurons_number), 0.0)
z = 0;
totalE = 1000
while (totalE > 0.001):
    for imageNb in range(1, 3):
        print("processing image: ", imageNb)
        t_vector = numpy.full(output_neurons_number, 0.2)
        t_vector[imageNb - 1] = 0.8
        imageNumber = '{:0>5}'.format(imageNb)

        for i in range(0, 1):

            # input_neurons, hidden_neurons, output_neurons = recognize(
            #     f"./learning_data/Sample00{imageNb}/img00{imageNb}-{imageNumber}.png",
            #     hidden_weights, output_weights)

            input_neurons, hidden_neurons, output_neurons = recognize(
                f"./learning_data/{imageNb-1}.png",
                hidden_weights, output_weights)

            b[imageNb-1] = output_neurons - t_vector

            print(z, ' image number:', imageNumber, 'image recognize ', imageNb - 1, ":", t_vector, output_neurons)
            z += 1

            for x in range(0, output_neurons_number):
                e_output[x] = b[imageNb-1][x] * output_neurons[x] * (1 - output_neurons[x]) * hidden_neurons

            for x in range(0, hidden_neurons_number):
                e_hidden[x] = 0
                for k in range(0, output_neurons_number):
                    e_hidden[x] += b[imageNb-1][k] * output_neurons[k] * (1 - output_neurons[k]) * output_weights[k, x]
                e_hidden[x] = e_hidden[x] * hidden_neurons[x] * (1 - hidden_neurons[x]) * input_neurons

        output_weights = output_weights - eta * e_output
        hidden_weights = hidden_weights - eta * e_hidden

    totalE = 0
    for i in range(1, output_neurons_number):
        totalE += 0.5 * numpy.sum(b[i - 1] * b[i - 1])
    print(totalE)

# ak, bk, ck = recognize("./learning_data/Sample001/img001-00001.png", hidden_weights, output_weights)
ak, bk, ck = recognize("./learning_data/0.png", hidden_weights, output_weights)
print(ck);
