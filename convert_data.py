import torch
import numpy as np

def convert_to_numpy(d):
    for key in d:
        # print(type(d[key]))
        if isinstance(d[key], torch.Tensor) :
            d[key] = d[key].cpu().detach().numpy()
        elif isinstance(d[key],torch.nn.parameter.Parameter):
            d[key] = d[key].cpu().detach().numpy()

        elif isinstance(d[key],dict):
            convert_to_numpy(d[key])
    return d
data = torch.load("unit.pth")
data = convert_to_numpy(data)
# breakpoint()
np.save('unit.npy',data)
# data = torch.load("snapshot_fw.dump")
# data = convert_to_numpy(data)
# # breakpoint()
# np.save('snapshot_fw.npy',data)