import argparse, os, sys
import numpy as np
import imageio
from scipy import ndimage
import torchvision
import torch
from torchvision.utils import save_image
from generateData import gen_batch

from AE import AE

from utils import get_interpolations

parser = argparse.ArgumentParser(
        description='Main function to call training for different AutoEncoders')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--model', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                    help='Which dataset to use')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

ae = AE(args)
architectures = {'AE':  ae}

print(args.model)
if __name__ == "__main__":
    try:
        os.stat(args.results_path)
    except :
        os.mkdir(args.results_path)

    try:
        autoenc = architectures[args.model]
    except KeyError:
        print('---------------------------------------------------------')
        print('Model architecture not supported. ', end='')
        print('Maybe you can implement it?')
        print('---------------------------------------------------------')
        sys.exit()

    try:
        for epoch in range(1, args.epochs + 1):
            autoenc.train(epoch)
            autoenc.test(epoch)
            if epoch % 10 == 0:
                torch.save(autoenc.state_dict(), "./weights/" + 'saved_model_{}.pt'.format(epoch))
                print('Model saved.')
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")

    with torch.no_grad():
        size = 64
        batch_size = 64
        gen_batch(batch_size)
        dir_names = [x for x in os.listdir(autoenc.model.path) if not ".png" in x]
        batch_size = 64 * 82
        images = torch.zeros(batch_size, 1, size, size)
        for i, dirname in enumerate(dir_names):
            filenames = [name for name in os.listdir(os.path.join(autoenc.model.path, dirname))]
            for j, filename in enumerate(filenames):
                images[i * 82 + j] = torchvision.io.read_image(os.path.join(autoenc.model.path, dirname, filename))[0, : , :]
        images = images.to(autoenc.device)
        images_per_row = 16
        interpolations = get_interpolations(args, autoenc.model, autoenc.device, images, images_per_row)

        sample = torch.randn(64, args.embedding_size).to(autoenc.device)
        sample = autoenc.model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                '{}/sample_{}_{}.png'.format(args.results_path, args.model, args.dataset))
        save_image(interpolations.view(-1, 1, 28, 28),
                '{}/interpolations_{}_{}.png'.format(args.results_path, args.model, args.dataset),  nrow=images_per_row)
        interpolations = interpolations.cpu()
        interpolations = np.reshape(interpolations.data.numpy(), (-1, 28, 28))
        interpolations = ndimage.zoom(interpolations, 5, order=1)
        interpolations *= 256
        imageio.mimsave('{}/animation_{}_{}.gif'.format(args.results_path, args.model, args.dataset), interpolations.astype(np.uint8))