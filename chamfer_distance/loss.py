import torch
from chamfer_distance import ChamferDistance as chamfer_dist
import time

p1 = torch.rand([10,25,3])
p2 = torch.rand([10,15,3])

s = time.time()
chd = chamfer_dist()
dist1, dist2, idx1, idx2 = chd(p1,p2)
loss = (torch.mean(dist1)) + (torch.mean(dist2))

torch.cuda.synchronize()
print(f"Time: {time.time() - s} seconds")
print(f"Loss: {loss}")