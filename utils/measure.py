import torch
import time

def time_run(model, input):
    torch.cuda.synchronize()
    start = time.time()
    output = model(input)
    torch.cuda.synchronize()
    end = time.time()
    return end - start