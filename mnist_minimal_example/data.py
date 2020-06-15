import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
import torchvision.datasets

batch_size = 256
data_mean = 0.128
data_std = 0.305

# amplitude for the noise augmentation
augm_sigma = 0.08
data_dir = 'mnist_data'
_default_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def unnormalize(x):
    '''go from normaized data x back to the original range'''
    return x * data_std + data_mean


def setup(device=_default_device,
          batchsize=batch_size,
          mean=data_mean,
          std=data_std,
          folder=data_dir,
          augm_std=augm_sigma):

    train_data = torchvision.datasets.MNIST(folder, train=True, download=True,
                                            transform=T.Compose([T.ToTensor(), lambda x: (x - mean) / std]))
    test_data = torchvision.datasets.MNIST(folder, train=False, download=True,
                                            transform=T.Compose([T.ToTensor(), lambda x: (x - mean) / std]))

    # Sample a fixed batch of 1024 validation examples
    val_x, val_l = zip(*list(train_data[i] for i in range(1024)))
    val_x = torch.stack(val_x, 0).to(device)
    val_l = torch.LongTensor(val_l).to(device)

    # Exclude the validation batch from the training data
    train_data.data = train_data.data[1024:]
    train_data.targets = train_data.targets[1024:]
    # Add the noise-augmentation to the (non-validation) training data:
    train_data.transform = T.Compose([train_data.transform, lambda x: x + augm_std * torch.randn_like(x)])

    train_loader  = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader   = DataLoader(test_data,  batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    return train_loader, test_loader, val_x, val_l
