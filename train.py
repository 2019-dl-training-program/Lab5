
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
import pickle
from dataloader import getTrainingData
from model import *
import csv

random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
parser.add_argument('--batchsize', type=int, default=256, help='size of the batches')
parser.add_argument('--lr_d', type=float, default=2e-4, help='initial learning rate of disciminator')
parser.add_argument('--lr_g', type=float, default=1e-3, help='initial learning rate of generator')
parser.add_argument('--output_str', type=str, default='./output', help='dir num of output')
parser.add_argument('--input_dim', type=int, default=62, help='input dimension of generator')
parser.add_argument('--num_dis_latent', type=int, default=1, help='number of discrete latent')
parser.add_argument('--dim_dis_latent', type=int, default=10, help='dimension of discrete latent')

opt = parser.parse_args()
print(opt)

os.makedirs(opt.output_str, exist_ok=True)

dataloader = getTrainingData(opt.batchsize)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Model Definition
netG = Generator(opt.input_dim).to(device)
netD = Discriminator().to(device)
netG.apply(weights_init)
netD.apply(weights_init)


# optimizer
# setup optimizer
optimizerD = optim.Adam(
    [{'params': netD.main.parameters()}, {'params': netD.D.parameters()}],
    lr=opt.lr_d,
    betas=(0.5, 0.999)
)
optimizerG = optim.Adam(
    [{'params': netG.parameters()}, {'params': netD.Q.parameters()}],
    lr=opt.lr_g,
    betas=(0.5, 0.999)
)


# training function
# use this function to generate noise
def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = torch.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0
    y_cat = y_cat.view(y.shape[0], num_columns, 1, 1)
    y_cat = y_cat.to(device)

    return y_cat


fix_noise = torch.randn([10, opt.input_dim-opt.dim_dis_latent, 1, 1], device=device).unsqueeze(1).repeat(1, 10, 1, 1, 1).view(100, opt.input_dim-opt.dim_dis_latent, 1, 1)
y = np.array([num for _ in range(10) for num in range(10)])
y_cat = to_categorical(y, opt.dim_dis_latent)
fix_noise = torch.cat([fix_noise, y_cat], dim=1)


# Loss for discrimination between real and fake images.
criterionD = nn.BCELoss()
# Loss for discrete latent code.
criterionQ = nn.CrossEntropyLoss()

with open(opt.output_str+'/loss.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(['err_D', 'err_G', 'err_Q'])
    
    
    for epoch in range(opt.n_epochs):
        netD.train()
        netG.train()
        
        # these are loss for disciminator, generator, and classifier
        # they can help you record the loss in a csv file
        errD_e, errG_e, errQ_e = 0, 0, 0
        
        for i, batch in enumerate(dataloader, 0):
            real = batch['x'].to(device)
            batch_size = real.shape[0]
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################




            ############################
            # (2) Update G & Q network: maximize log(D(G(z)))
            ###########################
 
            
            
            
            
            errD_e   += errD.item()
            errG_e   += errG.item()
            errQ_e   += errQ.item()

        errD_e /= len(dataloader)
        errG_e /= len(dataloader)
        errQ_e /= len(dataloader)
        writer.writerow([ errD_e, errG_e, errQ_e])
        csvfile.flush()

        print('[%d/%d] Loss_D: %.4f, Loss_G: %.4f, Loss_Q: %.4f'
                % (epoch, opt.n_epochs,
                   errD.item(), errG.item(), errQ.item()))
        # this help to save model
        torch.save({
                'netG': netG,
                'netD': netD
            }, os.path.join(opt.output_str, 'model_{}.pt'.format(epoch))
        )
        
        # this will help you show the result of your generator
        netD.eval()
        netG.eval()
        with torch.no_grad():
            fake = netG(fix_noise)
            
        vutils.save_image(fake.detach(), os.path.join(opt.output_str, 'result_{}.png'.format(epoch)), nrow=10)




        





