from model import *
import numpy as np
from torchvision import datasets, transforms, utils
import argparse

obs = (3, 16, 16)
def prepare():

    input_channels = obs[0]
    rescaling     = lambda x : (x - .5) * 2.
    rescaling_inv = lambda x : .5 * x  + .5
    resizer=transforms.Resize((16,16))
    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
    ds_transforms = transforms.Compose([resizer,transforms.ToTensor(), rescaling])
    temp_dataset=datasets.CIFAR10("./data", train=True, 
        download=True, transform=ds_transforms)

    train_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=16, shuffle=True, **kwargs)
    return train_loader
def simulate(train_loader,scale):
     
    
    #test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10("./data", train=False, 
    #                transform=ds_transforms), batch_size=16, shuffle=True, **kwargs)
    total_likelihood=0
    for batch_idx, (input,_) in enumerate(train_loader):
        total_likelihood+=simple_normal_likelihood(input,scale)
    return (total_likelihood/(len(train_loader)*np.prod(obs)*np.log(2.)))
        #print(input)
        #print(len(train_loader))
        #print(output.shape)
def main():  
    parser=argparse.ArgumentParser()
    parser.add_argument("-s","--scale",type=float,default=1.0)
    parser.add_argument('--cycle',action='store_true')
    args=parser.parse_args()
    if args.cycle:
        loader=prepare()
#        fout.write("Scale,Negative Log Likelihood\n")
        for i in np.linspace(0.1,500,2000):
            print(i)
            result=simulate(loader,i)
            with open("simple_models.csv",'a') as fout:
                fout.write("{}, {}\n".format(i,result.item()))
    else:
        loader=prepare()
        print(simulate(loader,args.scale))
main()
