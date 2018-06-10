import os


# for hidden in range (1000,1100,100):
#     path = f'./results/{hidden}/'
#     name = f'{hidden}_'
#     for i in range (1,6):
#         os.system(f"network.py --learn --input 1024 --hidden {hidden} --output 10 --path {path} --name {name}{i}")
#         os.system(f"python network.py --check --input 1024 --hidden {hidden} --output 10 --weights {path}{name}{i} --path {path} --name result_{name}{i}")

for x in range (1,11,1):
    eta = x/10
    path = f'./results/eta/{eta}/'
    name = f'{eta}_'
    for i in range (1,6):
        os.system(f"network.py --learn --input 1024 --hidden 600 --output 10 --path {path} --name {name}{i} --eta {eta}")
        os.system(f"python network.py --check --input 1024 --hidden 600 --output 10 --weights {path}{name}{i} --path {path} --name result_{name}{i}")