import os

#
#
# for hidden in range (700,1100,100):
#     path = f'./results_20/{hidden}/'
#     name = f'{hidden}_'
#     for i in range (1,6):
#         os.system(f"network.py --learn --input 1024 --hidden {hidden} --output 20 --path {path} --name {name}{i} --eta 0.1")
#         os.system(f"python network.py --check --input 1024 --hidden {hidden} --output 20 --weights {path}{name}{i} --path {path} --name result_{name}{i}")
#
#     # for x in range (1,4,1):
#     #     eta = x/10
#     #     path2 = f'./results_20/{hidden}/eta/{eta}/'
#     #     name2 = f'{eta}_'
#     #     for j in range (1,6):
#     #         os.system(f"network.py --learn --input 1024 --hidden {hidden} --output 20 --path {path2} --name {name2}{j} --eta {eta}")
#     #         os.system(f"python network.py --check --input 1024 --hidden {hidden} --output 20 --weights {path2}{name2}{j} --path {path2} --name result_{name2}{j}")

hidden = 700
for x in range(8, 10, 2):
    eta = x / 100
    path2 = f'./results_20/{hidden}/eta/{eta}/'
    name2 = f'{eta}_'
    for j in range(3, 6):
        os.system(
            f"network.py --learn --input 1024 --hidden {hidden} --output 20 --path {path2} --name {name2}{j} --eta {eta}")
        os.system(
            f"python network.py --check --input 1024 --hidden {hidden} --output 20 --weights {path2}{name2}{j} --path {path2} --name result_{name2}{j}")

eta =0.16
path2 = f'./results_20/{hidden}/eta/{eta}/'
name2 = f'{eta}_'
for j in range(3, 6):
    os.system(
        f"network.py --learn --input 1024 --hidden {hidden} --output 20 --path {path2} --name {name2}{j} --eta {eta}")
    os.system(
        f"python network.py --check --input 1024 --hidden {hidden} --output 20 --weights {path2}{name2}{j} --path {path2} --name result_{name2}{j}")
