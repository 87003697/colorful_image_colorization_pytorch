from tools.model import Model
from tools.data_generator import Preprocessor, ValDataset #train_gen, valid_gen
from tools.recorder import Loss_recorder
import numpy as np

import pdb
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from config import epoches, batch_size, learning_rate,beta_1, beta_2, weight_decay

import argparse
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from torchvision import transforms
import torchvision

import warnings
warnings.filterwarnings("ignore")



def train_net(args, model, device, train_loader, optimizer, epoch, weight):

    model.train()
    print('start training')
    
    loss_record = [] # record loss in batches
    
    for batch_idx, data in enumerate(tqdm.tqdm(train_loader)):
        
#         data, _ = data # This is for imagenet training dataset
        x, gt = data
        x, gt = x.to(device), gt.to(device)
        optimizer.zero_grad()
        y,_ = model(x)
        # back propogate
        loss = _loss(y, gt, weight)
        if torch.isnan(loss):
            pdb.set_trace()
        train_rec.take(loss / x.shape[0])
        loss.backward()
        optimizer.step()
        
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_rec.save()
            if args.dry_run:
                break
    return

def _loss(y, gt, weight):  
    # calculate loss
    loss_tmp = - gt * torch.log(y + 1e-10)
    loss_tmp_perpix = torch.sum(loss_tmp, axis = 1)
    max_idx_perpix = torch.argmax(gt, axis = 1) 
    prior_perpix = weight[max_idx_perpix.cpu()]
    prior_perpix = torch.tensor(prior_perpix).to(device)

    loss_perpix = prior_perpix * loss_tmp_perpix
    loss = torch.sum(loss_perpix) / (y.shape[2] * y.shape[3])
    
    return loss
    
    
def test_net(args, model, device, test_loader, weight):
    model.eval()
    test_loss = 0
    print('start validation')
    with torch.no_grad():
        for x, gt in tqdm.tqdm(test_loader):
            x, gt = x.to(device), gt.to(device)
            y, _ = model(x)
            test_loss += _loss(y, gt, weight).item()
            if args.dry_run:
                break
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(
    test_loss))
            
    
    return test_loss
                
train_rec = Loss_recorder('train')
val_rec = Loss_recorder('val')

                
                   
if __name__ == '__main__':
    
    # Add arguments
    parser = argparse.ArgumentParser(description='PyTorch Image Colorization')
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=epoches, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=learning_rate, metavar='LR',
                        help='learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--smoth-prior',type = str, default = 'data/prior_prob_smoothed.npy', help = 'the path to the smoothed prior distribution')
    parser.add_argument('--parellel',action = 'store_true', default = True, help = 'whether to apply parellecl computing')
    parser.add_argument('--resume',action = 'store_true', default = True, help = 'resume unfinished training')

    args = parser.parse_args()
    
    # specify device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # data loader configs
    train_path = 'data/coco/train2017'
    val_path = 'data/coco/val2017'
    
    kwargs = {'num_workers': torch.cuda.device_count() if args.parellel else 1,#args.num_workers, 
              'pin_memory': True} if args.parellel else {}
    
    transform = transforms.Compose([Preprocessor(),])
    
    # If you are using imagenet, uncomment this    
#     train_ds = torchvision.datasets.ImageFolder(train_path, 
#                                                  transform=transform)
    
    # If you are using coco, uncomment this
    train_ds = ValDataset(train_path, 
                        transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=args.batch_size * 
                                               torch.cuda.device_count() if args.parellel else args.batch_size,
                                               shuffle=True,
                                               **kwargs)
    
    val_ds = ValDataset(val_path, 
                        transform=transform)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                               batch_size=args.test_batch_size,
                                               shuffle=True,
                                             **kwargs)
    
    
    
    # build model
    model = Model().to(device)
    if args.parellel and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
        if args.resume:
            model.load_state_dict(torch.load("model.pt"))

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                          betas = (beta_1, beta_2),
                          weight_decay = weight_decay,
                          )
        
    # load prior distribution
    prior = np.load(args.smoth_prior)
    
    # calculate weight
    weight = 1/(0.5 * prior + 0.5 / 313)
    weight = weight / sum(prior * weight)
    
    recorder = Loss_recorder('val')            
        
    for epoch in range(1, args.epochs + 1):
        train_net(args, model, device, train_loader, optimizer, epoch, weight)

        loss = test_net(args, model, device, val_loader, weight)
        
        print('epoch finished')
        if args.save_model:
            torch.save(model.state_dict(), "model_shot3.pt")
            print('model is saved at', 'model.pt')
        
        # record loss
        val_rec.take(loss)        
        val_rec.save()

