import argparse
from generate_weights import generate_weights
from learn_network import LearnNetwork
from recognize_image import recognize

parser = argparse.ArgumentParser();

parser.add_argument("--weights", help="weights to detect the font")
parser.add_argument("--image", help="path to image to detect")
parser.add_argument("--input", help="input neurons number", type=int)
parser.add_argument("--output", help="output neurons number", type=int)
parser.add_argument("--hidden", help="hidden neurons number", type=int)
parser.add_argument("--learn", help="learn neural netowrk", action="store_true")


args = parser.parse_args()


if args.learn:
    if args.input != 0 and args.output !=0 and args.hidden != 0:
        hidden_weights, output_weights = generate_weights(args.input, args.hidden, args.output)
        learn = LearnNetwork(hidden_weights, output_weights, args.input, args.hidden, args.output)
        hw, ow =learn.learn_network()
        i = learn.getImage(learn.getImagePath(1, 1000))
        hn, on = recognize(i, args.hidden, args.output, hw, ow);
        print('result 0: ', on)

        i = learn.getImage(learn.getImagePath(2,1000))
        hn, on = recognize(i, args.hidden, args.output, hw, ow);
        print('result 1: ', on)
    else:
        print("Error")