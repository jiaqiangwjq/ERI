from torch.utils.data.dataset import Dataset, DataLoader
from torch.optim import Adam
import pickle
import os
import torch
from util import *
import torch
import torch.nn as nn
import argparse
from configs import *
from model import TE


torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.MSELoss()
Epoch = your_epoch


class Wav2Vec_Dataset(Dataset):
    def __init__(self, mode):
        super(hubert_Dataset, self).__init__()
        self.base_path = 'PATH' 
        self.label_path = "PATH"
        with open(self.base_path, 'rb') as f:
            self.features = pickle.load(f)
        with open(self.label_path, 'rb') as f:
            self.labels = pickle.load(f)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx] 


def Train(epoch):
    train_dataset = Wav2Vec_Dataset(mode="train")
    val_dataset = Wav2Vec_Dataset(mode="val")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    lr = your_lr
    model = TE(opts, opts.d_model * opts.modal_num)
    optimizer = Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    
    # train
    t_preds = []
    t_labels = []
    best_pcc = -1.0
    for i in range(epoch):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            output = model(x)
            if len(output.shape) == 1:
                output = output.unsqueeze(dim=0)
            t_preds.append(output.cpu().detach())
            t_labels.append(y.cpu().detach())
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        tt_preds = torch.cat(t_preds, dim=0)
        tt_labels = torch.cat(t_labels, dim=0)
        pcc_train = calu_PCC(tt_preds.numpy(), tt_labels.numpy())[1]

        # eval
        model.eval()
        with torch.no_grad():
            v_preds = []
            v_labels = []
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                if len(output.shape) == 1:
                    output = output.unsqueeze(dim=0)
                v_preds.append(output.cpu().detach())
                v_labels.append(y.cpu().detach())
        vv_preds = torch.cat(v_preds, dim=0)
        vv_labels = torch.cat(v_labels, dim=0)
        pcc_val = calu_PCC(vv_preds.numpy(), vv_labels.numpy())[1]
        if pcc_val > best_pcc:
            best_pcc = pcc_val

if __name__ == "__main__":
    Train(Epoch)
