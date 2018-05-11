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

    def __init__(self, hidden_weights, output_weights,input_neurons_number, hidden_neurons_number, output_neurons_number):
        self.input_neurons_number = input_neurons_number
        self.hidden_neurons_number = hidden_neurons_number
        self.output_neurons_number = output_neurons_number

        self.hidden_weights = hidden_weights
        self.output_weights = output_weights


    def getImagePath(self, imageNb, image):
        formatedNb = '{:0>3}'.format(imageNb)
        formatedImage = '{:0>5}'.format(image)
        if self.input_neurons_number == 49:
            return f"./learning_data/{imageNb-1}.png"
        elif self.input_neurons_number == 4096:
            return f"./learning_data/Sample{formatedNb}_small/Sample{formatedNb}/img{formatedNb}-{formatedImage}.png"
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

    def check_network(self):
        totalImages = 0;
        totalCorrect = 0;
        for image in range(801, 1017):
            for imageNb in range(1, self.output_neurons_number + 1):
                i = self.getImage(self.getImagePath(imageNb, image))
                hn, on = recognize(i, self.hidden_neurons_number, self.output_neurons_number, self.hidden_weights, self.output_weights)
                t_vector = self.setExpectedResult(imageNb - 1)

                b = on - t_vector;
                totalError = 0.5 * numpy.sum(b * b)
                totalImages = totalImages + 1
                if totalError < 0.1:
                    totalCorrect = totalCorrect + 1;
                print(f'image {image} result {imageNb-1}: ', on, 'expected', t_vector,'whole error: ', totalError )

        print('total images: ', totalImages, ' total good detected: ', totalCorrect, ' % correct: ', (totalCorrect/totalImages) * 100, '%')