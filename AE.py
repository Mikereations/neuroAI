import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import os
import numpy as np
from generateData import gen_batch
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append('./')
from AE_utils import FC_Encoder, FC_Decoder, CNN_Encoder, CNN_Decoder

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        output_size = args.embedding_size
        self.encoder = CNN_Encoder(output_size)
        self.image_size = 64
        self.path = os.path.join("./", "inputs")
        # calculated the class imbalance imprically from the data. class 1 occurs 0.0118 times as often as class 0
        self.weights = torch.tensor([0.0118, 1.0])
        convert_tensor = transforms.ToTensor()
        # initialize the loss function with the class imbalance weights
        self.BCE = torch.nn.BCEWithLogitsLoss(pos_weight=self.weights)
        self.decoder = CNN_Decoder(args.embedding_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x.view(-1, self.image_size * self.image_size))
        return self.decode(z)

class AE(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.model = Network(args)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def loss_function(self, recon_x, x):
        loss = self.BCE(recon_x, x.view(-1, self.model.image_size ** 2), reduction='sum')
        return loss

    def train(self, epoch, batch_size=64):
        self.model.train()
        train_loss = 0
        for batch_idx in range(100):
            gen_batch(batch_size)
            dir_names = [x for x in os.listdir(self.model.path) if not ".png" in x and "data" in x]
            filenum = [len(os.listdir(os.path.join(self.model.path, dirname))) for dirname in dir_names]
            # print(np.sum(filenum))
            batch_size = np.sum(filenum)
            data = torch.zeros(batch_size, 1, 64, 64)
            k = 0
            batch_size = 64
            for i, dirname in enumerate(dir_names):
                filenames = [name for name in os.listdir(os.path.join(self.model.path, dirname))]
                for j, filename in enumerate(filenames):
                    data[k] = torchvision.io.read_image(os.path.join(self.model.path, dirname, filename))[0, : , :]/255
                    k += 1
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch = self.model(data)
            # print(data[0])
            loss = self.loss_function(recon_batch, data)
            loss.backward()
            # if batch_idx % 10 == 0:
            #     # plt.imshow(data[13].view(64, 64).detach().numpy(), cmap='gray')
            #     # plt.show()
            #     positive = (data[13].view(-1).numpy() == 1)
            #     print(data[13].view(-1).numpy()[positive])
            #     print(recon_batch[13].view(-1).detach().numpy()[positive])
            # print("loss: ", loss.item())
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), 100 * len(data),
                    float(batch_idx),
                    loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / 100 * len(data)))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i in range(10):
                gen_batch(64)
                dir_names = [x for x in os.listdir(self.model.path) if not ".png" in x and "data" in x]
                filenum = [len(os.listdir(os.path.join(self.model.path, dirname))) for dirname in dir_names]
                # print(np.sum(filenum))
                batch_size = np.sum(filenum)
                data = torch.zeros(batch_size, 1, 64, 64)
                k = 0
                batch_size = 64
                for i, dirname in enumerate(dir_names):
                    filenames = [name for name in os.listdir(os.path.join(self.model.path, dirname))]
                    for j, filename in enumerate(filenames):
                        data[k] = torchvision.io.read_image(os.path.join(self.model.path, dirname, filename))[0, : , :]/255
                        k += 1
                data = data.to(self.device)
                recon_batch = self.model(data)
                test_loss += self.loss_function(recon_batch, data).item()

        test_loss /= 10 * 64
        print('====> Test set loss: {:.4f}'.format(test_loss))