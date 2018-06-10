import csv

from scipy import misc
import numpy
from recognize_image import recognize

class CheckNetwork:
    input_neurons = [];
    hidden_weights = [];
    output_weights = [];
    input_neurons_number = 0;
    hidden_neurons_number = 0;
    output_neurons_number = 0;
    eta = 0.3
    beta = 1
    networkError = 0;
    epoqueErrors = [];
    good_number = [];
    bad_number = [];

    def __init__(self, hidden_weights, output_weights,input_neurons_number, hidden_neurons_number, output_neurons_number):
        self.input_neurons_number = input_neurons_number
        self.hidden_neurons_number = hidden_neurons_number
        self.output_neurons_number = output_neurons_number

        self.hidden_weights = hidden_weights
        self.output_weights = output_weights

        self.good_number = numpy.full(self.output_neurons_number, 0)
        self.bad_number = numpy.full(self.output_neurons_number, 0)


    def getImagePath(self, imageNb, image):
        formatedNb = '{:0>3}'.format(imageNb)
        formatedImage = '{:0>5}'.format(image)
        if self.input_neurons_number == 49:
            return f"./learning_data/{imageNb-1}.png"
        elif self.input_neurons_number == 1024:
            return f"./learning_data/small/Sample{formatedNb}/img{formatedNb}-{formatedImage}.png"
        else:
            return f"./learning_data/Sample{formatedNb}/img{formatedNb}-{formatedImage}.png"

    def setExpectedResult(self, imageNb):
        t_vector = numpy.full(self.output_neurons_number, 0.1)
        t_vector[imageNb] = 0.9
        return t_vector;

    def getImage(self, path):
        image = misc.imread(path, True, "L")
        arr = numpy.array(image.flatten() / 255)
        return numpy.insert(arr, 0,1)

    def check_network(self, path, filename):
        totalImages = 0;
        totalCorrect = 0;
        for image in range(801, 1017):
            for imageNb in range(1, self.output_neurons_number + 1):
                i = self.getImage(self.getImagePath(imageNb, image))
                hn, on = recognize(i, self.hidden_neurons_number, self.output_neurons_number, self.hidden_weights, self.output_weights)
                t_vector = self.setExpectedResult(imageNb - 1)

                b = on - t_vector;

                totalImages = totalImages + 1
                if numpy.argmax(on) == (imageNb - 1):
                    totalCorrect = totalCorrect + 1;
                    self.good_number[imageNb-1] =  self.good_number[imageNb-1] + 1;
                else:
                    self.bad_number[imageNb - 1] = self.bad_number[imageNb - 1] + 1;
                    print(f'image {image} result {imageNb-1}: ', on, 'expected', t_vector, 'arg max',numpy.argmax(on) )

        with open(f'{path}{filename}.csv', 'w') as f_output:
            csv_output = csv.writer(f_output, lineterminator='\n', delimiter=',', dialect="excel")
            csv_output.writerow([totalImages, totalCorrect, (totalCorrect/totalImages) * 100])
            csv_output.writerow(self.good_number)
            csv_output.writerow(self.bad_number)
            csv_output.writerow(self.good_number/(self.good_number + self.bad_number))

        print('total images: ', totalImages, ' total good detected: ', totalCorrect, ' % correct: ', (totalCorrect/totalImages) * 100, '%')
        print('good number', self.good_number, ' bad number', self.bad_number, 'percent good', self.good_number/(self.good_number + self.bad_number))
