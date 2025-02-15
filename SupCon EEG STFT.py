import torch
import torch.nn as nn
from torch import Tensor
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import math
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from torchsummary import summary
import gc
import seaborn as sns


########################################################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        
    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, (1, 1))
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

class TripleLaneCNN(nn.Module):
    def __init__(self):
        super(TripleLaneCNN, self).__init__()
        
        self.temporal_cnn = nn.Sequential(
            ResidualBlock(1, 16, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(16, 32, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(32, 64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SqueezeExcitation(64)
        )
        
        self.high_freq_cnn = nn.Sequential(
            ResidualBlock(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SqueezeExcitation(64)
        )
        
        self.low_freq_cnn = nn.Sequential(
            ResidualBlock(1, 16, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(16, 32, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(32, 64, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SqueezeExcitation(64)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
    def forward(self, x):
        temporal_out = self.temporal_cnn(x)
        
        high_freq_out = self.high_freq_cnn(x)
        
        low_freq_out = self.low_freq_cnn(x)
        
        temporal_out = self.global_avg_pool(temporal_out)
        high_freq_out = self.global_avg_pool(high_freq_out)
        low_freq_out = self.global_avg_pool(low_freq_out)
        
        temporal_out = temporal_out.view(temporal_out.size(0), -1)
        high_freq_out = high_freq_out.view(high_freq_out.size(0), -1)
        low_freq_out = low_freq_out.view(low_freq_out.size(0), -1)
        
        combined = torch.cat((temporal_out, high_freq_out, low_freq_out), dim=1)
        embedding = self.fc(combined)
        
        return embedding
   
# ***************************************************************************************************
# ***************************************************************************************************
# ***************************************************************************************************
# ***************************************************************************************************
def load_checkpoint(model, optimizer, folder_name, file_name):
    '''
        inputs:
            model: nn.Module model
            optimizer: torch.optim optimizer
            folder_name: folder name to load the checkpoint model
            file_name: the name of checkpoint file 
        ---------------------------------------------------------
        output:
            loads the checkpoint and updates model and optimizer 
    '''
    checkpoint = torch.load(folder_name +'\\'+ file_name)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def save_checkpoint(state, folder_name, filename='checkpoint.pth.tar'):
    '''
        inputs:
            state: a dictionary of information that we want to be saved,
                   "state_dict" and "optimizer" should be included in order
                   for the model to be loaded. "state_dict" is model.state_dict()
                   and "optimizer" is optimizer.state_dict().
                   
                   - it should look something like this:
                       {
                        'lr': 0.000000015
                        'epoch': 5,
                        'state_dict': sim_model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                       }
        --------------------------------------------------------------------------
          folder_name: self explainatory
          filename: .tar file name
        ---------------------------------------------------------------------------
        output:
            saves the model and any other info we want to be attached to the saved file.
            
    '''
    torch.save(state, folder_name + '\\' + filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
            print('learning rate was set to '+str(cur_lr))


def plot_losses(trainLoss, validLoss):
    losses = pd.DataFrame(trainLoss, columns=['Train Loss'])
    losses['Validation Loss'] = validLoss
    losses = losses.dropna()
    losses.plot()



def save_loss_database(connection, cur, values):
    '''
    Table Columns:
        Model
        Epoch
        TrainLoss
        ValidationLoss
        SleepLoss
        ApneaLoss
        InClassDistTrain
        BetweenClassDistTrain
        ReconstructionLossTrain
        InClassDistValid
        BetweenClassDistValid
        ReconstructionLossValid

    '''
    cur.execute('DELETE FROM Loss WHERE Model = "'+values[0]+'" AND Epoch = '+str(values[1]))
    cur.execute("INSERT INTO Loss (Model, Epoch, TrainLoss, ValidationLoss, SleepLossTrain, ApneaLossTrain, InClassDistTrain, BetweenClassDistTrain, ReconstructionLossTrain, InClassDistValid, BetweenClassDistValid, ReconstructionLossValid) VALUES " + str(values))
    connection.commit()


def train_one_epoch(train_loader, valid_loader, model, criterion, optimizer, epoch, lossLog, valid_loss_log, lossLogEpoch, validLossLogEpoch, dbconnection, dbcur, folder_name, log_freq=1000):
    '''
        info: function that trains the model for one epoch and logs its performance in terms of 
              training time, data loading time and training loss and saves a checkpoint every epoch.
        -------------------------------------------------------------------------------------------
        input:
            train_loader: dataLoader
            model: nn.Module model
            criterion: loss function
            optimizer: torch.optim
            epoch: current epoch for logging 
            log_freq: how often to log
        -------------------------------------------------------------------------------------------
        !NOTE! make adjustments for different models in lines with #needs adjustment tags
        !NOTE! if batch size of training and validation are the same, we can plot their loss together and compare them
        
    '''
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
                            len(train_loader),
                            [batch_time, data_time, losses],
                            prefix="Epoch: [{}]".format(epoch)
                            )


    model.train()

    end = time.time()
    trainLoss = 0
    for i, d in enumerate(tqdm(train_loader)):
        
        data_time.update(time.time() - end)
        
        data = d[0].to('cuda')
        target = d[1].to('cuda')
        
        output = model(data)
        loss = criterion(output, target)

        losses.update(loss.item(), data.size(0)) 

        lossLog.append(loss.item()) 
        trainLoss += loss.item()
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_freq == 0:
            progress.display(i)
    trainLoss /= len(train_loader)
    gc.collect()

    
    model.eval()  
    valLoss = 0
    for data, target in valid_loader:
        data = data.to('cuda')
        target = target.to('cuda')
        
        output = model(data)
        loss = criterion(output, target)
        valLoss += loss.item()
        
        valid_loss_log.append(loss.item())
    valLoss /= len(valid_loader)

    lossLogEpoch.append(trainLoss) 
    validLossLogEpoch.append(valLoss)
    gc.collect()
    save_loss_database(dbconnection, dbcur, (folder_name.split("\\")[-1], epoch+1, trainLoss, valLoss, '', '', '', '', '', '','','') )
            
            
            
def train(model, criterion, optimizer, folder_name, checkpoint_prefix, starting_epoch, epochs, lr, dbconnection, dbcur, datafolder, log_freq=1000):
    '''
        info: trains the model for "epochs" and saves a checkpoint every epoch
        inputs:
            model: nn.Module model
            trainloader: train dataLoader
            criterion: loss function
            optimizer: torch.optim
            folder_name: folder name to save the model checkpoint
            checkpoint_prefix: checkpoint file name's prefix (can add model specific info like lr, model name, ...)
    '''
    lossLog = []
    validLossLog = []
    lossLogEpoch = []
    validLossLoggEpoch = []
    trainlist = os.listdir('D:\\Data\\cfs\\Full\\EEG STFT Dataset\\Train')
    testlist = os.listdir('D:\\Data\\cfs\\Full\\EEG STFT Dataset\\Test')
    trainbase = 'D:\\Data\\cfs\\Full\\EEG STFT Dataset\\Train\\'
    testbase = 'D:\\Data\\cfs\\Full\\EEG STFT Dataset\\Test\\'
    lossLog = []
    validLossLog = []
    lossLogEpoch = []
    validLossLoggEpoch = []
    trainind = 0
    testind = 0
    for epoch in range(starting_epoch, epochs):
        # load trainer and tester - each csv contains 100,000 samples 
        trainer = SampleLoader(file=trainbase + trainlist[trainind])
        trainloader = DataLoader(dataset=trainer, batch_size=256, shuffle=True, num_workers=6)
        tester = SampleLoader(file=testbase + testlist[testind])
        validationloader = DataLoader(dataset=tester, batch_size=256, shuffle=True, num_workers=6)
        trainind += 1
        testind += 1
        if trainind > len(trainlist) - 1:
            trainind = 0
        if testind > len(testlist) - 1:
            testind = 0
        #print(len(trainloader))
        # adjust learning rate
        adjust_learning_rate(optimizer, lr, epoch)
        
        # train for one epoch
        train_one_epoch(trainloader, validationloader, model, criterion, optimizer, epoch, lossLog, validLossLog,lossLogEpoch, validLossLoggEpoch,dbconnection, dbcur,folder_name, log_freq)

        # make adjustments
        save_checkpoint({ 
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            
        }, folder_name, filename=checkpoint_prefix+'- epoch -'+str(epoch)+'.pth.tar')
        gc.collect()





# ****************************************************************
# ****************************************************************
class SampleLoader(Dataset):
    '''
        this is a dataloader for a siamese model so we need three images as input and we don't have labels
        (Anchor sample, Positive sample with the same class, negative sample with a different class)
    '''
    def __init__(self, file):
        super().__init__()
        self.x = torch.tensor(np.load(file)['x'])
        self.y = torch.tensor(np.load(file)['y'])
        #self.dataset = torch.tensor(np.load(file))
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        
        img = torch.tensor(self.x[idx]).unsqueeze(0).type(torch.float32)
        img = img.reshape((1,120, 60))
        labels = self.y[idx]
        return (img, torch.tensor(labels).unsqueeze(0))




#################################################################################################################
###########################################Loss######################################################################
#################################################################################################################


class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, class_weights=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.class_weights = class_weights

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)

        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        exp_similarity = torch.exp(similarity)

        labels = labels.view(-1, 1).long() 
        same_label_mask = (labels == labels.T).float()  
        exclude_self_mask = 1 - torch.eye(labels.size(0), device=labels.device)  
        positive_mask = same_label_mask * exclude_self_mask

        denominator = exp_similarity * exclude_self_mask
        denominator = denominator.sum(dim=1, keepdim=True) + 1e-10  

        numerator = (exp_similarity * positive_mask).sum(dim=1) + 1e-10

        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(labels.device)
            weights = self.class_weights[labels.squeeze().long()]
            loss_per_sample = -torch.log(numerator / denominator) * weights
        else:
            loss_per_sample = -torch.log(numerator / denominator)

        loss = loss_per_sample.mean()
        loss = torch.clamp(loss, min=0.0)
        return loss





#################################################################################################################
#################################################################################################################
#################################################################################################################

def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripleLaneCNN().to(device)
    model.apply(weights_init_normal)
    summary(model, (1, 120, 60))

    criterion= SupConLoss().to(device)
    lr = 0.0002

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    params = filter(lambda p: p.requires_grad, model.parameters())
    print("number of parameters for autoencoder model is {}.".format(sum([np.prod(p.size()) for p in params])))
    
    
    import sqlite3 
    con = sqlite3.connect("C:\\Users\\m.afshari\\Desktop\\Masters Project Codes\\Loss\\Loss.db")
    cur = con.cursor()
    
   
    load_checkpoint(model,optimizer,'C:\\Users\\m.afshari\\Desktop\Masters Project Codes\\Masters Project Model Checkpoints\\SupCon EEG STFT 2', 'Checkpoint- epoch -46.pth.tar')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('training started: ')
    train(
    model=model, 
      criterion=criterion, 
      optimizer=optimizer, 
      folder_name= 'C:\\Users\\m.afshari\\Desktop\Masters Project Codes\\Masters Project Model Checkpoints\\SupCon EEG STFT 2',
      checkpoint_prefix= 'Checkpoint',
      starting_epoch=47,
      epochs=200,
      lr=lr,
      dbconnection=con,
      dbcur=cur,
      datafolder='D:\\Data\\sleep-edf-database-expanded-1.0.0\\Data\\78 STFT',
    log_freq=10
    )
