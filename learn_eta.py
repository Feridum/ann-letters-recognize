import os;

hidden = 700

eta = 0.12
path2 = f'./results_20/{hidden}/eta/{eta}/'
name2 = f'{eta}_'
for j in range(4, 6):
    os.system(
        f"network.py --learn --input 1024 --hidden {hidden} --output 20 --path {path2} --name {name2}{j} --eta {eta}")
    os.system(
        f"python network.py --check --input 1024 --hidden {hidden} --output 20 --weights {path2}{name2}{j} --path {path2} --name result_{name2}{j}")

for x in range(14, 16, 2):
    eta = x / 100
    path2 = f'./results_20/{hidden}/eta/{eta}/'
    name2 = f'{eta}_'
    for j in range(1, 6):
        os.system(
            f"network.py --learn --input 1024 --hidden {hidden} --output 20 --path {path2} --name {name2}{j} --eta {eta}")
        os.system(
            f"python network.py --check --input 1024 --hidden {hidden} --output 20 --weights {path2}{name2}{j} --path {path2} --name result_{name2}{j}")
