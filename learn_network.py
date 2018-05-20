from scipy import misc
import numpy
from recognize_image import recognize
from random import randint

class LearnNetwork:
    input_neurons = [];
    hidden_neurons = [];
    output_neurons = [];
    hidden_weights = [];
    output_weights = [];
    input_neurons_number = 0;
    hidden_neurons_number = 0;
    output_neurons_number = 0;
    eta = 0.1
    beta = 1
    networkError = 0;
    epoqueErrors = [];

    def __init__(self, hidden_weights, output_weights,input_neurons_number, hidden_neurons_number, output_neurons_number):
        self.input_neurons_number = input_neurons_number
        self.hidden_neurons_number = hidden_neurons_number
        self.output_neurons_number = output_neurons_number

        self.hidden_weights = hidden_weights
        self.output_weights = output_weights

        self.epoqueErrors = numpy.full(5, 0.2)


    def calculateTotalNetworkError(self, b):
        self.networkError += numpy.sum(b * b) / self.output_neurons_number

    def calcuteError(self, b):
        e_output = numpy.full((self.output_neurons_number, self.hidden_neurons_number+1), 0.0)
        e_hidden = numpy.full((self.hidden_neurons_number, self.input_neurons_number+1), 0.0)

        for x in range(0, self.output_neurons_number):
            e_output[x] = self.beta * b[x] * self.output_neurons[x] * (1 - self.output_neurons[x]) * self.hidden_neurons

        for x in range(0, self.hidden_neurons_number):
            e_hidden[x] = 0
            for k in range(0, self.output_neurons_number):
                e_hidden[x] += b[k] * self.output_neurons[k] * (1 - self.output_neurons[k]) * self.output_weights[k, x]
            e_hidden[x] = self.beta * e_hidden[x] * self.hidden_neurons[x] * (1 - self.hidden_neurons[x]) * self.input_neurons


        return e_hidden, e_output


    def setExpectedResult(self, imageNb):
        t_vector = numpy.full(self.output_neurons_number,0.1)
        t_vector[imageNb] = 0.9
        return t_vector;

    def getImagePath(self, imageNb, image):
        formatedNb = '{:0>3}'.format(imageNb)
        formatedImage = '{:0>5}'.format(image)
        if self.input_neurons_number == 49:
            return f"./learning_data/{imageNb-1}.png"
        elif self.input_neurons_number == 1024:
            return f"./learning_data/small/Sample{formatedNb}/img{formatedNb}-{formatedImage}.png"
        else:
            return f"./learning_data/Sample{formatedNb}/img{formatedNb}-{formatedImage}.png"

    def getImage(self, path):
        image = misc.imread(path, True, "L")
        arr = numpy.array(image.flatten() / 255)
        return numpy.insert(arr, 0,1)
        # return arr

    def learn_network(self):

        isEnough = False;
        i = 0;
        while not isEnough:
            self.networkError = 0
            for imageNb in range(1, self.output_neurons_number+1):

                if i <= 800:
                    image = randint(1, 800)
                else:
                    image = i % 800 + 1

                self.input_neurons = self.getImage(self.getImagePath(imageNb, image))
                self.hidden_neurons, self.output_neurons = recognize(self.input_neurons, self.hidden_neurons_number, self.output_neurons_number, self.hidden_weights, self.output_weights)
                t_vector = self.setExpectedResult(imageNb-1)

                b = self.output_neurons - t_vector
                e_hidden, e_output = self.calcuteError(b)

                self.output_weights = self.output_weights - (self.eta * e_output)
                self.hidden_weights = self.hidden_weights - (self.eta * e_hidden)
                self.calculateTotalNetworkError(b)
                print(i, ' image number:', imageNb, "expected result:", t_vector, "result: ",self.output_neurons, 'error', b)

            self.epoqueErrors[i%5] = self.networkError
            isEnough = self.networkError < 0.1 or (round(self.epoqueErrors[0],8) == round(self.epoqueErrors[1],8) == round(self.epoqueErrors[2],8) == round(self.epoqueErrors[3],8) == round(self.epoqueErrors[4],8))
            print('epoque errors', self.epoqueErrors, 'is Enough', isEnough)
            i+=1
            self.eta  = self.eta - 0.00001;
        return self.hidden_weights, self.output_weights