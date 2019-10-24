#!/usr/bin/env python
# coding: utf-8

# Import statements

# In[1]:


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


# This is the function we use to produce our distributions for the gaussian dataset

# In[2]:


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


# Command Line argument parsing code

# In[5]:


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
parser.add_argument('-f','--few_shot',type=int,default=0,nargs='?',help='How many examples to use in training')
parser.add_argument('--separate_constants',action='store_true',help='Whether to separate the different constant lines or not')

args = parser.parse_args()


# To ensure reproducibility

# In[7]:


# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)


# Model naming, and file name creation

# In[9]:


model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}_nr-data{}'.format(args.lr, args.nr_resnet, args.nr_filters,args.few_shot)
assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)
writer = SummaryWriter(log_dir=os.path.join('runs', model_name))


# Data Pipeline

# In[10]:


sample_batch_size = 15
obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 16, 16)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
resizer=transforms.Resize((16,16))
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([resizer,transforms.ToTensor(), rescaling])


# Setting up the loss operations and datasets depending on dataset we chose in the command line arguments

# In[11]:


if 'mnist' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                        train=True, transform=ds_transforms), batch_size=args.batch_size, 
                            shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

elif 'cifar' in args.dataset : 
    temp_dataset=datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms)
    if args.few_shot !=0:
        #temp_dataset=torch.utils.data.random_split(temp_dataset,args.few_shot)[0]
        temp_dataset=torch.utils.data.Subset(temp_dataset,np.random.choice(range(len(temp_dataset)),args.few_shot,replace=False))

    train_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))


# This code creates the dataset of images of constant pixel value

# In[12]:

if not args.separate_constants:
    len_constant=args.batch_size
    images=[torch.ones([1,3,16,16])*(-1+2*i/(len_constant-1)) for i in range(len_constant)]
    constant_images=torch.cat(images)
else:
    len_constant=args.batch_size
    images=[torch.ones([1,3,16,16])*(-1+2*i/(len_constant-1)) for i in range(len_constant)]
    constant_image_list=images
# This code creates the low-variance images

# In[13]:


len_low_variance=args.batch_size
X=get_truncated_normal(mean=0,sd=0.05,low=-1,upp=1)
lowvar_images=torch.reshape(torch.as_tensor(X.rvs(len_low_variance*256*3),dtype=torch.float),(len_low_variance,3,16,16))

model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
model = model.cuda()


# This code creates the SVHN images

# In[15]:

x=datasets.SVHN(root=args.data_dir,split='train',transform=ds_transforms,download=True)
train_loader_svhn=torch.utils.data.DataLoader(x,batch_size=args.batch_size,shuffle=True)
image_batch=next(iter(train_loader_svhn))[0]
print(image_batch.shape)
if args.load_params:
    load_part_of_model(model, args.load_params)
    # model.load_state_dict(torch.load(args.load_params))
    print('model parameters loaded')
    

# In[16]:


optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)


# This is the method that we will use to sample from the pixelCNN

# In[17]:


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


# In[ ]:


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
    if args.separate_constants:
        constant_input=[x.cuda() for x in constant_image_list]
        constant_var=[Variable(y) for y in constant_input]
        output=[model(y) for y in constant_var]
        loss=[loss_op(x,y) for x, y in zip(constant_var,output)]
    else:
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
    
    svhn_input=image_batch.cuda()
    svhn_var=Variable(svhn_input)
    output_svhn=model(svhn_var)
    loss_svhn=loss_op(svhn_var,output_svhn)
    averaged_svhn_loss=loss_svhn/(image_batch.shape[0]*np.prod(obs)*np.log(2.))
    print("SVHN likelihood: {}".format(averaged_svhn_loss))
    if not args.separate_constants: 
        with open('resultsNew fewShot{}.csv'.format(args.few_shot),'a') as f:
            f.write("{},{},{},{},{},{}\n".format(epoch,
global_loss/deno_train,
test_loss/deno, 
loss/(constant_images.shape[0]*np.prod(obs)*np.log(2.)),
loss_lowvar/(lowvar_images.shape[0]*np.prod(obs)*np.log(2.)),
averaged_svhn_loss
))
    else:
        
        with open('resultsNew fewShot{}.csv'.format(args.few_shot),'a') as f:
            f.write("{},{},{},{},{},{}\n".format(epoch,
global_loss/deno_train,
test_loss/deno, 
",".join([str(y/(np.prod(obs)*np.log(2.))) for y in loss]),
loss_lowvar/(lowvar_images.shape[0]*np.prod(obs)*np.log(2.)),
averaged_svhn_loss
))
    if (epoch + 1) % args.save_interval == 0: 
        torch.save(model.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
        #print('sampling...')
        #sample_t = sample(model)
        #sample_t = rescaling_inv(sample_t)
        #utils.save_image(sample_t,'images/{}_{}.png'.format(model_name, epoch), 
        #       nrow=5, padding=0)


# In[ ]:




