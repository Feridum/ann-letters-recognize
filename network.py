import argparse
import numpy
import datetime
from generate_weights import generate_weights
from learn_network import LearnNetwork
from recognize_image import recognize
from check_network import CheckNetwork

parser = argparse.ArgumentParser();

parser.add_argument("--weights", help="weights to detect the font")
parser.add_argument("--image", help="path to image to detect")
parser.add_argument("--input", help="input neurons number", type=int)
parser.add_argument("--output", help="output neurons number", type=int)
parser.add_argument("--hidden", help="hidden neurons number", type=int)
parser.add_argument("--learn", help="learn neural netowrk", action="store_true")
parser.add_argument("--check", help="check neural netowrk weights", action="store_true")
parser.add_argument("--path", help="path for file save")
parser.add_argument("--name", help="file name")
parser.add_argument("--eta", help="file name", type=float)
parser.add_argument("--acc", help="acc", type=float)

args = parser.parse_args()

if args.learn:
    if args.input != 0 and args.output !=0 and args.hidden != 0:
        if args.eta == None or args.eta == '':
            eta = 0.1
        else:
            eta = args.eta

        if args.acc == None or args.acc == '':
            acc = 0.1
        else:
            acc = args.acc

        if args.weights == None or args.weights == '' :
            hidden_weights, output_weights = generate_weights(args.input, args.hidden, args.output)
        else:
            hidden_weights = numpy.genfromtxt(f"{args.weights}_hidden.csv", delimiter=',')
            output_weights = numpy.genfromtxt(f"{args.weights}_output.csv", delimiter=',')

        learn = LearnNetwork(hidden_weights, output_weights, args.input, args.hidden, args.output, eta, acc)
        success, hw, ow, errors =learn.learn_network()

        if success:
            if args.name == '':
                name = int(round(datetime.datetime.now().timestamp()))
            else:
                name = args.name

            if args.path == '':
                path = './'
            else:
                path = args.path

            numpy.savetxt(f"{path}{name}_hidden.csv", hw, delimiter=",")
            numpy.savetxt(f"{path}{name}_output.csv", ow, delimiter=",")
            numpy.savetxt(f"{path}{name}_errors.csv", errors, delimiter=",")

            print('Success ')
        else:
            print('Too much epoques')
    else:
        print("Error")
elif args.check:
    if args.check != '' and args.input != 0 and args.output !=0 and args.hidden != 0:
        if args.name == '':
            name = int(round(datetime.datetime.now().timestamp()))
        else:
            name = args.name

        if args.path == '':
            path = './'
        else:
            path = args.path

        hw = numpy.genfromtxt(f"{args.weights}_hidden.csv", delimiter=',')
        ow = numpy.genfromtxt(f"{args.weights}_output.csv", delimiter=',')
        check = CheckNetwork(hw, ow, args.input, args.hidden, args.output)
        check.check_network(path, name);
    else:
        print("error")