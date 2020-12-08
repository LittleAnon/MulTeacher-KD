from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/atts')
import numpy as np; np.random.seed(0)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import torch
d = torch.load('att.bin', map_location='cpu')
step = 0
for s, t in zip(d['s'], d['t']):
    for index, s_layer in enumerate(s):
        fig = plt.figure()
        ax = plt.imshow(torch.mean(s_layer, dim=0).detach().numpy())
        writer.add_figure(f's_att_{index}', fig, step)
    for index, t_layer in enumerate(t):
        fig = plt.figure()
        ax = plt.imshow(torch.mean(t_layer, dim=0).detach().numpy())
        writer.add_figure(f't_att_{index}', fig, step)
    
writer.close()