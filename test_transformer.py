import argparse, os, sys
from transformer import GPTLanguageModel
from AE import AE
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from generateData import gen_batch
from results_vis import compare_results
from torchvision.utils import save_image


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

# load the weights of both models
for model in architectures :
    if model == "AE" :
        architectures[model].model.load_state_dict(torch.load(os.path.join(args.results_path, model + '.pt'), map_location=torch.device('cpu')))
        for param in architectures[model].model.parameters():
            param.requires_grad = False
        
    else :
        architectures[model].load_state_dict(torch.load(os.path.join(args.results_path, model + '.pt')))
        # architectures[model]
    

def performance() :
    total_distance = []
    total_pixels = []
    number_of_glimpses = []
    losses = np.zeros(19)
    overlap = []
    for i in range(10000) :
        # 1. get a batch of images 
        batch, count = gen_batch(64)
        # 2. use the policy to generate 20 positions for each image
        positions = [architectures["Transformer"].policy() if i % 20 != 0 else np.array([0,0]) for i in range(64 * 20)]
        dp = np.array([positions[i] - positions[i - 1] for i in range(64 * 20) if i % 20 != 0], dtype=np.float32)
        dp = torch.from_numpy(dp).reshape(64, 19, 2)
        # 3. use the positions to get the next image in the sequence
        image_data = [batch[int(i / 20)][int(positions[i][0]):int(positions[i][0]+64), int(positions[i][1]):int(positions[i][1]+64)] for i in range(64 * 20)]
        input_image_data = torch.stack([torch.from_numpy(image_data[i])/255 for i in range(64 * 20) if i % 20 != 19]).reshape(64 * 19, 1 , 64, 64)
        output_image_data = torch.stack([torch.from_numpy(image_data[i])/255 for i in range(64 * 20) if i % 20 != 0]).reshape(64 * 19, 1 , 64, 64)
        # 4. encode the images using the encoder
        # print(input_image_data.dtype, input_image_data.shape)
        input_encoded_images = architectures["AE"].model.encode(input_image_data.to(architectures["AE"].device))
        output_encoded_images = architectures["AE"].model.encode(output_image_data.to(architectures["AE"].device))
        output_decoded_images = architectures["AE"].model(output_image_data.to(architectures["AE"].device))
        save_image(output_decoded_images.view(64 * 19, 1, 64, 64) * 255,
                '{}/sampleout_{}_{}.png'.format(args.results_path, args.model, args.dataset))
        x, y = input_encoded_images.view(-1, 19, 32).to(architectures["AE"].device), output_encoded_images.view(-1, 19, 32).to(architectures["AE"].device)
        dp = dp.to(architectures["AE"].device)
        # For calculating the overlapping pixels
        psudeo_image = torch.zeros(64, 640, 640)
        # psudeo_image[:, int(positions[:,:, 0]):int(positions[:,:,0] + 64), int(positions[:,:, 1]):int(positions[:,:, 1] + 64)] = 1.0 
        for j in range(19) :
            # calculate the evlauation metrics
            # get the data for the first j glimpses
            xj, yj, dpj = x[:, :j + 1, :].view(64, j + 1, 32) ,y[:, j, :], dp[:, :j + 1, :].view(64, j + 1, 2)
            # for image in psudeo_image :
                
            # store the current number of glimpses
            #number_of_glimpses.append(j * xj.shape[0])
            # store the total distances
            #total_distance.append(dpj[:, :, 0].abs().item()**2 + dpj[:, :, 1].abs().item()**2)
            # store the total number of overlapping pixels
            #psudeo_image = torch.zeros(64, 640, 640)
            #for k in range(j + 1) :
            #    psudeo_image[:, int(positions[:,k, 0]):int(positions[:,k,0] + 64), int(positions[:,k, 1]):int(positions[:,k, 1] + 64)] = 1.0
            #overlap.append(64 * 64 * xj.shape[1] - torch.sum(psudeo_image, axis = 0).item())
            # store the total number of white pixels
            #psudeo_image = torch.zeros(64, 640, 640)

            # 6. compute the distance between the original image and the reconstructed image
            #total_distance.append(torch.mean(torch.abs(x - y)).item())
            # 7. calculate the error on the predicted images
            predicted, ـ = architectures["Transformer"](xj, dpj)
            logits = predicted[:, -1, :]
            predicted_image = architectures["Transformer"].model.model.decode(logits.view(-1, 32))
            ground_truth = architectures["Transformer"].model.model.decode(yj.view(-1, 32))
            # save_image(predicted_image.view(64, 1, 64, 64) * 255,
                # '{}/predicted_{}_{}.png'.format(args.results_path, args.model, args.dataset))
            # save_image(ground_truth.view(64, 1, 64, 64) * 255,
                # '{}/gt_{}_{}.png'.format(args.results_path, args.model, args.dataset))
            # compare_results(predicted_image.reshape(-1, 64, 64), ground_truth.reshape(-1, 64, 64))
            losses[j] += architectures["Transformer"].model.loss_function(predicted_image, ground_truth)
            print("current loss at j = {} is {}".format(j, losses[j] / (64 * (i + 1))))
    return x, y


def test_performance() :
    total_distance = []
    total_pixels = []
    number_of_glimpses = []
    losses = np.zeros(19)
    overlap = []
    # 1. get a batch of images 
    batch, count = gen_batch(64)
    # 2. use the policy to generate 20 positions for each image
    positions = [architectures["Transformer"].policy() if i % 20 != 0 else np.array([0,0]) for i in range(64 * 20)]
    dp = np.array([positions[i] - positions[i - 1] for i in range(64 * 20) if i % 20 != 0], dtype=np.float32)
    dp = torch.from_numpy(dp).reshape(64, 19, 2)
    # 3. use the positions to get the next image in the sequence
    image_data = [batch[int(i / 20)][int(positions[i][0]):int(positions[i][0]+64), int(positions[i][1]):int(positions[i][1]+64)] for i in range(64 * 20)]
    input_image_data = torch.stack([torch.from_numpy(image_data[i])/255 for i in range(64 * 20) if i % 20 != 19]).reshape(64 * 19, 1 , 64, 64)
    output_image_data = torch.stack([torch.from_numpy(image_data[i])/255 for i in range(64 * 20) if i % 20 != 0]).reshape(64 * 19, 1 , 64, 64)
        
    return output_image_data



if __name__ == "__main__":
    performance()
