import argparse, os, sys
import numpy as np
import torchvision
import time 
import torch
from torchvision.utils import save_image, make_grid
from generateData import gen_batch
from AE import AE
from transformer import GPTLanguageModel
from transformer_non import GPTLanguageModel as GPTLanguageModel_non
from transformer_typical import GPTLanguageModel as GPTLanguageModel_typical
from full_transformer_auto import GPTLanguageModel as fullTransformer


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
parser.add_argument('--weights_path', type=str, default='weights_full/', metavar='N',
                    help='Where to store images')
parser.add_argument('--model', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                    help='Which dataset to use')
parser.add_argument('--method', type=str, default='itt', metavar='N',
                    help='Whether to train on an itterative approach or not')


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 2 # what is the maximum context length for predictions?
max_iters = 50000
eval_interval = 1
learning_rate = 1e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256
n_head = 2
n_layer = 2
dropout = 0.2
# ------------

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)


# Initialize the model and load its weights
autoenc = AE(args)

transformer = GPTLanguageModel(autoenc).to(autoenc.device) if args.method == "itt" else GPTLanguageModel_non(autoenc).to(autoenc.device)
if args.method == "typical" :
    transformer = GPTLanguageModel_typical(autoenc).to(autoenc.device)
elif args.method == "itt" : 
    transformer = fullTransformer(args).to(autoenc.device)

print("The method is : ", args.method)
architectures = {'AE':  autoenc, 'Transformer': transformer}
if __name__ == "__main__":

    try:
        weight_path = os.listdir("./weights_full")
        weight_path = [w for w in weight_path if w.startswith("saved_model")]
        if len(weight_path) > 0:
            autoenc.model.load_state_dict(torch.load("./weights_full/" + weight_path[-1]))
            print('Model loaded : {}'.format(weight_path[-1]))
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")

    def instant_test(architectures, model, epoch) :
        total_distance = []
        total_pixels = []
        number_of_glimpses = []
        losses = np.zeros(19)
        overlap = []
        for i in range(1) :
            xj, yj, dpj = gen_data(64)
            predicted_image, ـ = model(xj, dpj)
            predicted_image = architectures["AE"].model.decode(predicted_image.view(-1, 32))
            ground_truth = architectures["AE"].model.decode(yj.view(-1, 32))
            image_to_be_saved = make_grid(predicted_image.view(64 * 19, 1, 64, 64) * 255, 19)
            ground_saved = make_grid(ground_truth.view(64 * 19, 1, 64, 64) * 255, 19)
            save_image(image_to_be_saved,
                '{}/predicted_{}_{}.png'.format(args.results_path, args.model, epoch))
            save_image(ground_saved,
                '{}/gt_{}_{}.png'.format(args.results_path, args.model, epoch))

    # prepare the data for the transformer, we have images in sequence and their positions.
    # we need to create a batch of images and a batch of positions
    # we also need to create a batch of the next images in the sequence
    def gen_data(batch_size=256):
        # 1. get a batch of images 
        num_examples = batch_size
        batch, count = gen_batch(num_examples)
        # 2. use the policy to generate 20 positions for each image
        positions = architectures["Transformer"].selective_policy(batch)
        #[architectures["Transformer"].policy() if i % 20 != 0 else np.array([0,0]) for i in range(64 * 20)]
        dp = [positions[i] - positions[i - 1] for i in range(num_examples * 20) if i % 20 != 0]
        dp.insert(0, positions[0])
        dp = np.array(dp, dtype=np.float32)
        dp = torch.from_numpy(dp).reshape(num_examples, 19, 2)
        positions = np.array(positions, dtype=np.float32)
        # 3. use the positions to get the next image in the sequence
        image_data = [batch[int(i / 20)][int(positions[i][0]):int(positions[i][0]+64), int(positions[i][1]):int(positions[i][1]+64)]/255.0 for i in range(num_examples * 20)]
        input_image_data = torch.stack([torch.from_numpy(image_data[i]) for i in range(num_examples * 20) if i % 20 != 19]).reshape(num_examples * 19, 1 , 64, 64)
        output_image_data = torch.stack([torch.from_numpy(image_data[i]) for i in range(num_examples * 20) if i % 20 != 0]).reshape(num_examples * 19, 1 , 64, 64)
        # 4. encode the images using the encoder
        # print(input_image_data.dtype, input_image_data.shape)
        input_encoded_images = architectures["AE"].model.encode(input_image_data.to(architectures["AE"].device))
        output_encoded_images = architectures["AE"].model.encode(output_image_data.to(architectures["AE"].device))
        x, y = input_encoded_images.reshape(-1, 19, 32).to(architectures["AE"].device), output_encoded_images.reshape(-1, 19, 32).to(architectures["AE"].device)
        if args.method == "itt":
            return x, y, dp.to(architectures["AE"].device)
        elif args.method == "full" : 
            positions = torch.from_numpy(positions).reshape(num_examples, 20, 2).to(architectures["AE"].device)
            return input_image_data.to(architectures["AE"].device), output_image_data.to(architectures["AE"].device), positions
        else :
            positions = torch.from_numpy(positions).reshape(num_examples, 20, 2)[:, 1:, :].to(architectures["AE"].device)
            return x, y, positions
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model = architectures["Transformer"]
        # model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(1)
            for k in range(1):
                X, Y, dp = gen_data()
                # print(dp.dtype, dp.shape)
                _, loss = model(X, dp, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        # model.train()
        return out
    
    @torch.no_grad()
    def estimate_loss_once():
        out = {}
        model = architectures["Transformer"]
        # model.eval()
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
                loss = architectures["Transformer"].weighted_loss_function(recon_batch, data)
                losses = loss.item()
                # print("The current loss is : " , losses)
            out[split] = losses
        # model.train()
        return out
    
    model = architectures["Transformer"]
    epoch = 0
    try:
        weight_path = os.listdir(args.weights_path)
        weight_path = [w for w in weight_path if w.startswith("transformer")]
        
        if len(weight_path) > 0:
            print("Loading weights", weight_path[-1])
            model.load_state_dict(torch.load(args.weights_path + weight_path[-1], map_location=autoenc.device))
            print('The trained transformer model was loaded succesfully, weights : {}'.format(weight_path[-1]))
            print("Resuming training")
            epoch = int(weight_path[-1].split("_")[1].split(".")[0])
        else : 
            print("No available weights, training from scratch")
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")
    m = model.to(architectures["AE"].device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lagtime = 0
    xb, yb, dp = gen_data()
    for iter in range(epoch, max_iters + epoch):
        t1 = time.time()
        # every once in a while evaluate the loss on train and val sets
        
        
        if iter % 500 == 0 or iter == max_iters - 1 :
            # losses = estimate_loss()
            # print(f"step {iter}: train loss {losses['train']:.10f}, val loss {losses['val']:.10f}, time {lagtime:.3f}")
            instant_test(architectures,model, iter)
            torch.save(m.state_dict(), args.weights_path + 'transformer_{:2d}.pt'.format(iter))
            print('Model saved.')
        # sample a batch of data
        
        xb, yb, dp = gen_data()

        # evaluate the loss
        logits, loss = model(xb, dp, yb, architectures["AE"])
        # print("The shape of the recon batch is : ", recon_batch.shape, " and the shape of the data is : ", data.shape)
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = loss.item()
            print(f"step {iter}: train loss {losses:.10f}, time {lagtime:.3f}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lagtime = time.time() - t1
