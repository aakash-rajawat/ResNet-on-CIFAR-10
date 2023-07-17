import torch

if torch.cuda.is_available():
    print("it is")
else:
    print("it is not")
