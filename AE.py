import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import os
from generateData import gen_batch
from torchvision import datasets, transforms
import torchvision
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
        convert_tensor = transforms.ToTensor()
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
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.model.image_size ** 2), reduction='sum')
        return BCE

    def train(self, epoch, batch_size=64):
        self.model.train()
        train_loss = 0
        for batch_idx in range(100000):
            gen_batch(batch_size)
            filenames = [name for name in os.listdir(self.model.path)]
            data = torch.zeros(batch_size, 1, 64, 64)
            for i, filename in enumerate(filenames):
                data[i] = torchvision.io.read_image(os.path.join(self.model.path, filename))[0, : , :]
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch = self.model(data)
            loss = self.loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * 64, 64 * self.args.log_interval,
                    100. * batch_idx / 64 * self.args.log_interval,
                    loss.item() / 64))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i in range(100):
                gen_batch(64)
                filenames = [name for name in os.listdir(self.path)]
                data = torch.zeros(64, 1, 64, 64)
                for j, filename in enumerate(filenames):
                    data[j] = torchvision.io.read_image(os.path.join(self.path, filename))
                data = data.to(self.device)
                recon_batch = self.model(data)
                test_loss += self.loss_function(recon_batch, data).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))