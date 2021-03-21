from time import time

from tqdm import tqdm
import torch
import torch.nn
import torch.optim
import numpy as np

import model
import data
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cinn = model.MNIST_cINN(5e-4)
cinn.to(DEVICE)
scheduler = torch.optim.lr_scheduler.MultiStepLR(cinn.optimizer, milestones=[20, 40], gamma=0.1)

N_epochs = 60
t_start = time()
nll_mean = []

print('Epoch\tBatch/Total \tTime \tNLL train\tNLL val\tLR')
trainld, testld, val_x, val_l = data.setup(DEVICE)

for epoch in range(N_epochs):
    for i, (x, l) in enumerate(trainld):
        x, l = x.to(DEVICE), l.to(DEVICE)
        z, log_j = cinn(x, l)

        nll = torch.mean(z**2) / 2 - torch.mean(log_j) / model.ndim_total
        nll.backward()
        torch.nn.utils.clip_grad_norm_(cinn.trainable_parameters, 10.)
        nll_mean.append(nll.item())
        cinn.optimizer.step()
        cinn.optimizer.zero_grad()

        if not i % 50:
            with torch.no_grad():
                z, log_j = cinn(val_x, val_l)
                nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / model.ndim_total

            print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (epoch,
                                                            i, len(trainld),
                                                            (time() - t_start)/60.,
                                                            np.mean(nll_mean),
                                                            nll_val.item(),
                                                            cinn.optimizer.param_groups[0]['lr'],
                                                            ), flush=True)
            nll_mean = []
    scheduler.step()

torch.save(cinn.state_dict(), 'output/mnist_cinn.pt')
