import os

weights = './better/initial2'
hidden = 600

acc = 0.025
path = f'./better/{hidden}/{acc}/'

for i in range(1, 6):
    for x in range(4, 12, 2):
        eta = x / 100
    name = f'eta_{eta}'
    os.system(
        f"network.py --learn --input 1024 --hidden {hidden} --output 10 --path {path} --name {name}{i} --eta {eta} --acc {acc} --weights {weights} ")
    os.system(
        f"python network.py --check --input 1024 --hidden {hidden} --output 10 --weights {path}{name}{i} --path {path} --name result_{name}{i} --acc {acc}")
