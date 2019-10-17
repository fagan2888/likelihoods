import time
import pdb
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter
from utils import * 
from model import * 
from PIL import Image
import numpy
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='models',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be either cifar|mnist')
parser.add_argument('-p', '--print_every', type=int, default=50,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=5,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0002, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=32,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
args = parser.parse_known_args()[0]

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(args.lr, args.nr_resnet, args.nr_filters)
assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)
writer = SummaryWriter(log_dir=os.path.join('runs', model_name))

sample_batch_size = 15
obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 16, 16)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
resizer=transforms.Resize((16,16))
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([resizer,transforms.ToTensor(), rescaling])

if 'mnist' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                        train=True, transform=ds_transforms), batch_size=args.batch_size, 
                            shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

elif 'cifar' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))
"""
We will now create the dataset of images of constant pixel value
"""
len_constant=32
images=[torch.ones([1,3,16,16])*(-1+2*i/(len_constant-1)) for i in range(len_constant)]
constant_images=torch.cat(images)
"""
We will now create the low variance gaussian images
"""
len_low_variance=32
X=get_truncated_normal(mean=0,sd=0.05,low=-1,upp=1)
lowvar_images=torch.reshape(torch.as_tensor(X.rvs(len_low_variance*256*3),dtype=torch.float),(len_low_variance,3,16,16))

model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
model = model.cuda()
"""
We will now create SVHN images
"""
train_loader_svhn=torch.utils.data.DataLoader(datasets.SVHN
if args.load_params:
    load_part_of_model(model, args.load_params)
    # model.load_state_dict(torch.load(args.load_params))
    print('model parameters loaded')

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

def sample(model):
    model.train(False)
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    #data=torch.nn.functional.interpolate(data,scale_factor=0.5)
    for i in range(obs[1]//2):
        for j in range(obs[2]//2):
            data_v = Variable(data, volatile=True)
            out   = model(data_v, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

print('starting training')
writes = 0
global_loss=0
for epoch in range(args.max_epochs):
    model.train(True)
    torch.cuda.synchronize()
    train_loss = 0.
    time_ = time.time()
    model.train()
    global_loss=0
    for batch_idx, (input,_) in enumerate(train_loader):
        input = input.cuda()
        #print(input)
        #print(len(train_loader))
        input = Variable(input)
        output = model(input)
        #print(output.shape)
        loss = loss_op(input, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()
        global_loss+=loss.data.item()
        if (batch_idx +1) % args.print_every == 0 : 
            deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
            writer.add_scalar('train/bpd', (train_loss / deno), writes)
            print('loss : {:.4f}, time : {:.4f} progress: {}/{}'.format(
                (train_loss / deno), 
                (time.time() - time_),batch_idx, len(train_loader)))
            train_loss = 0.
            writes += 1
            time_ = time.time()
        #if batch_idx>48:
           # break
    # decrease learning rate
    scheduler.step()
    deno_train=batch_idx*args.batch_size*np.prod(obs)*np.log(2.) 
    torch.cuda.synchronize()
    model.eval()
    test_loss = 0.
    for batch_idx, (input,_) in enumerate(test_loader):
        input = input.cuda()
        #print(type(input))
        input_var = Variable(input)
        output = model(input_var)
        loss = loss_op(input_var, output)
        test_loss += loss.data.item()
        del loss, output
    
    deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
    writer.add_scalar('test/bpd', (test_loss / deno), writes)
    print('test loss : %s' % (test_loss / deno))
    print(global_loss/len(train_loader.dataset))
    #pdb.set_trace()
    constant_input=constant_images.cuda()
    constant_var=Variable(constant_input)
    output=model(constant_var)
    loss=loss_op(constant_var,output)
    print("Constant iamge likelihood: {}".format(loss/(constant_images.shape[0]*np.prod(obs)*np.log(2.))))
    #with open('results.csv'):
    


    lowvar_input=lowvar_images.cuda()
    lowvar_var=Variable(lowvar_input)
    output_lowvar=model(lowvar_var)
    loss_lowvar=loss_op(lowvar_var,output_lowvar)
    print("Low var image likelihood: {}".format(loss_lowvar/(lowvar_images.shape[0]*np.prod(obs)*np.log(2.))))
    with open('results.csv','a') as f:
        f.write("{},{},{},{},{}".format(epoch,global_loss/deno_train,test_loss/deno, loss/(constant_images.shape[0]*np.prod(obs)*np.log(2,)),loss_lowvar/(lowvar_images.shape[0]*np.prod(obs)*np.log(2,))))
    
    if (epoch + 1) % args.save_interval == 0: 
        torch.save(model.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
        #print('sampling...')
        #sample_t = sample(model)
        #sample_t = rescaling_inv(sample_t)
        #utils.save_image(sample_t,'images/{}_{}.png'.format(model_name, epoch), 
        #       nrow=5, padding=0)
