import argparse, os, sys
import numpy as np
import imageio
from scipy import ndimage
import torchvision
import torch
from torchvision.utils import save_image
from generateData import gen_batch
from test_transformer import test_performance

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
        weight_path = os.listdir("./weights")
        weight_path = [w for w in weight_path if w.startswith("saved_model")]
        epoch = 1
        train = False
        if len(weight_path) > 0:
            autoenc.model.load_state_dict(torch.load("./weights/" + weight_path[-1], map_location=torch.device('cpu')))
            print('Model loaded : {}'.format(weight_path[-1]))
            epoch = int(weight_path[-1].split("_")[2].split(".")[0]) + 1
        if train == True:
            for epoch in range(epoch, epoch + args.epochs + 1):
                autoenc.train(epoch)
                autoenc.test(epoch)
                if epoch % 10 == 0:
                    torch.save(autoenc.model.state_dict(), "./weights/" + 'saved_model_{:2d}.pt'.format(epoch))
                    print('Model saved.')
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")

    with torch.no_grad():
        size = 64
        batch_size = 64
        batch, count = gen_batch(batch_size)
        data = torch.zeros(count, 1, 64, 64)
        keta = 0
        batch_size = 64
        for i,b in enumerate(batch):
            flag = False
            for j in range(0, 640 - 63, 64):
                for k in range(0, 640 - 63, 64):
                    if np.sum(b[j:j+64, k:k+64]) > 0:
                        data[keta] = torch.from_numpy(b[j:j+64, k:k+64])/255
                        keta += 1
                    elif flag == False :
                        data[keta] = torch.from_numpy(b[j:j+64, k:k+64])/255
                        keta += 1
                        flag = True
        size = data[0].shape[2]  
        data = test_performance()
        images = data.to(autoenc.device)
        samples = autoenc.model(images)
        images_per_row = 16
        # interpolations = get_interpolations(args, autoenc.model, autoenc.device, images, images_per_row)
        # sample = torch.randn(count, args.embedding_size).to(autoenc.device)
        # sample = autoenc.model.decode(sample).cpu()
        save_image(images.view(64 * 19, 1, size, size) * 255,
                '{}/originals_{}_{}.png'.format(args.results_path, args.model, args.dataset))
        save_image(samples.view(64 * 19, 1, size, size) * 255,
                '{}/sample_{}_{}.png'.format(args.results_path, args.model, args.dataset))
        # save_image(interpolations.view(-1, 1, size, size),
                # '{}/interpolations_{}_{}.png'.format(args.results_path, args.model, args.dataset),  nrow=images_per_row)
        # interpolations = interpolations.cpu()
        # interpolations = np.reshape(interpolations.data.numpy(), (-1, size, size))
        # interpolations = ndimage.zoom(interpolations, 5, order=1)
        # interpolations *= 256
        # imageio.mimsave('{}/animation_{}_{}.gif'.format(args.results_path, args.model, args.dataset), interpolations.astype(np.uint8))