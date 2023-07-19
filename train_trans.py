import argparse, os, sys
import numpy as np
import imageio
from scipy import ndimage
import torchvision
import torch
from torchvision.utils import save_image
from generateData import gen_batch
from AE import AE
from transformer import GPTLanguageModel


parser = argparse.ArgumentParser(
        description='Main function to call training for a transformer AutoEncoders')
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


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 2 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)


# Initialize the model and load its weights
autoenc = AE(args)
transformer = GPTLanguageModel(autoenc).to(autoenc.device)
architectures = {'AE':  autoenc, 'Transformer': transformer}
if __name__ == "__main__":
    try:
        weight_path = os.listdir("./weights")
        weight_path = [w for w in weight_path if w.startswith("saved_model")]
        epoch = 1
        if len(weight_path) > 0:
            architectures["AE"].model.load_state_dict(torch.load("./weights/" + weight_path[-1], map_location=autoenc.device))
            print('The trained auto encoder model was loaded succesfully, weights : {}'.format(weight_path[-1]))
            epoch = int(weight_path[-1].split("_")[2].split(".")[0]) + 1
            architectures["AE"].model.eval()
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")

    # prepare the data for the transformer, we have images in sequence and their positions.
    # we need to create a batch of images and a batch of positions
    # we also need to create a batch of the next images in the sequence
    def gen_data():
        # 1. get a batch of images 
        batch, count = gen_batch(64)
        # 2. use the policy to generate 20 positions for each image
        positions = [architectures["Transformer"].policy() if i % 20 != 0 else np.array([0,0]) for i in range(64 * 20)]
        dp = np.array([positions[i] - positions[i - 1] for i in range(64 * 20) if i % 20 != 0], dtype=np.float32)
        dp = torch.from_numpy(dp).reshape(64, 19, 2)
        # 3. use the positions to get the next image in the sequence
        image_data = [batch[int(i / 20)][int(positions[i][0]):int(positions[i][0]+64), int(positions[i][1]):int(positions[i][1]+64)]/255.0 for i in range(64 * 20)]
        input_image_data = torch.stack([torch.from_numpy(image_data[i]) for i in range(64 * 20) if i % 20 != 19]).reshape(64 * 19, 1 , 64, 64)
        output_image_data = torch.stack([torch.from_numpy(image_data[i]) for i in range(64 * 20) if i % 20 != 0]).reshape(64 * 19, 1 , 64, 64)
        # 4. encode the images using the encoder
        # print(input_image_data.dtype, input_image_data.shape)
        input_encoded_images = architectures["AE"].model.encode(input_image_data.to(architectures["AE"].device))
        output_encoded_images = architectures["AE"].model.encode(output_image_data.to(architectures["AE"].device))
        x, y = input_encoded_images.reshape(-1, 19, 32).to(architectures["AE"].device), output_encoded_images.reshape(-1, 19, 32).to(architectures["AE"].device)
        return x, y, dp.to(architectures["AE"].device)
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model = architectures["Transformer"]
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(10)
            for k in range(10):
                X, Y, dp = gen_data()
                # print(dp.dtype, dp.shape)
                logits, _ = model(X, dp, Y)
                recon_batch = architectures["AE"].decoder(logits)
                data = architectures["AE"].model.decode(Y)
                loss = architectures["AE"].loss_function(recon_batch, data)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    @torch.no_grad()
    def estimate_loss_once():
        out = {}
        model = architectures["Transformer"]
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(1)
            for k in range(1):
                X, Y, dp = gen_data()
                # print("The shape of Y is : ", Y.shape)
                # print(dp.dtype, dp.shape)
                logits, _ = model(X, dp, Y)
                recon_batch = architectures["AE"].model.decode(logits.view(-1, 32))
                data = architectures["AE"].model.decode(Y.view(-1, 32))
                # print("The shape of the recon batch is : ", recon_batch.shape, " and the shape of the data is : ", data.shape)
                loss = architectures["AE"].loss_function(recon_batch, data)
                losses = loss.item()
                # print("The current loss is : " , losses)
            out[split] = losses
        model.train()
        return out
    
    model = architectures["Transformer"]
    epoch = 0
    try:
        weight_path = os.listdir("./weights")
        weight_path = [w for w in weight_path if w.startswith("transformer")]
        
        if len(weight_path) > 0:
            print("Loading weights", weight_path[-1])
            model.load_state_dict(torch.load("./weights/" + weight_path[-1], map_location=autoenc.device))
            print('The trained transformer model was loaded succesfully, weights : {}'.format(weight_path[-1]))
            print("Resuming training")
            epoch = int(weight_path[-1].split("_")[2].split(".")[0]) + 1
        else : 
            print("No available weights, training from scratch")
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")
    m = model.to(architectures["AE"].device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(epoch, max_iters + epoch):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss_once()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            torch.save(m.state_dict(), "./weights/" + 'transformer_{:2d}.pt'.format(iter))
            print('Model saved.')
        # sample a batch of data
        xb, yb, dp = gen_data()

        # evaluate the loss
        logits, loss = model(xb, dp, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
