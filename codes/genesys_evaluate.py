import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy as sp
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.linalg as linalg
import re
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, silhouette_score

device = 'cpu'

## Class to prepare mini atlas as training batch 
class Root_Dataset(Dataset):
    def __init__(self, features, labels):
        super(Root_Dataset).__init__()
        self.features = features
        self.labels = labels
        self.celltype = np.array([i.split('_', 1)[0] for i in self.labels], dtype='int')
        self.timebin = np.array([i.split('_', 1)[1] for i in self.labels], dtype='float32')
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        idx = int(np.random.choice(range(10),1))
        cell = np.copy(self.features[np.random.choice(np.where(self.celltype==-1)[0], 1), :])
        cell = cell.astype(np.float32)
        
        
        cell_list = []
        cell_list.append(cell.reshape(-1))
        for k in range(1,11):
                sample = np.copy(self.features[np.random.choice(np.where((self.celltype==idx) & (self.timebin==k))[0], 1), :]).astype(np.float32).reshape(-1)
                cell_list.append(sample)
                
        cell_seq = {}
        cell_seq['x']=np.array(cell_list)
        cell_seq['y']=idx
        return cell_seq

## If the data has no QCs
class Root_Dataset_NoQC(Dataset):
    def __init__(self, features, labels):
        super(Root_Dataset).__init__()
        self.features = features
        self.labels = labels
        self.celltype = np.array([i.split('_', 1)[0] for i in self.labels], dtype='int')
        self.timebin = np.array([i.split('_', 1)[1] for i in self.labels], dtype='float32')
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        idx = int(np.random.choice(range(10),1))
        cell = np.array(pd.Series(0.0, index=np.arange(17513)))
        cell = cell.astype(np.float32)
        
        
        cell_list = []
        cell_list.append(cell.reshape(-1))
        for k in range(1,11):
                #sample = np.copy(self.features[np.random.choice(np.where((self.celltype==idx) & (self.timebin==(11-k)))[0], 1), :]).astype(np.float32).reshape(-1) # Reverse time bin: 10,9,8,7,6,...,1
                sample = np.copy(self.features[np.random.choice(np.where((self.celltype==idx) & (self.timebin==(k)))[0], 1), :]).astype(np.float32).reshape(-1)
                cell_list.append(sample)

        #cell_list.append(cell.reshape(-1))        
        cell_seq = {}
        cell_seq['x']=np.array(cell_list)
        cell_seq['y']=idx
        return cell_seq

## Define regularizer
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.
    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.to(x.device).repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x

## Define D operation in TD-VAE's paper 
class DBlock(nn.Module):
    """ A basic building block for parametralize a normal distribution.
    It is corresponding to the D operation in the reference Appendix.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)
        
    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = t * torch.sigmoid(self.fc2(input))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma

class Decoder(nn.Module):
    """ The decoder layer converting state to observation.
    Because the observation is MNIST image whose elements are values
    between 0 and 1, the output of this layer are probabilities of 
    elements being 1.
    """
    def __init__(self, z_size, hidden_size, x_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)
        
    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p
    
#class Decoder(nn.Module):
#    """ The decoder layer converting state to observation.
#    """
#    def __init__(self, z_size, hidden_size, x_size):
#        super(Decoder, self).__init__()
#        self.layers = nn.Sequential(
#        nn.ReLU(inplace=False),
#        nn.Linear(z_size, hidden_size),
#        nn.ReLU(inplace=False),
#        nn.Linear(hidden_size, x_size),    
#       )
#    def forward(self, input):
#        return self.layers(input)

## Define Genesys model
class ClassifierLSTM(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.2):
        super(ClassifierLSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, embedding_dim),
            nn.Dropout(p=drop_prob),
            GaussianNoise(sigma=0.2),
        )

        self.fc = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim*2, output_size),
        )
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)
        
        ## From belief to state (b to z)
        ## this is corresponding to P_B distribution in the reference
        self.b_to_z = DBlock(embedding_dim*2, hidden_dim, embedding_dim*2) 

        ## Given belief and state at time t2, infer the state at time t1
        ## infer state
        self.bz2_infer_z1 = DBlock(embedding_dim*2 + embedding_dim*2, hidden_dim, embedding_dim*2) 

        ## Given the state at time t1, model state at time t2 through state transition
        ## state transition
        self.z1_to_z2 = DBlock(embedding_dim*2, hidden_dim, embedding_dim*2)

        ## state to observation
        self.z_to_x = Decoder(embedding_dim*2, hidden_dim, input_size)
        
        #Initialize the layers 
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    torch.nn.init.constant_(m.bias, 0)
        
    def predict(self, x, hidden, t):
        #batch_size = x.size(0)
        #hidden = tuple([each.repeat(1, batch_size, 1).data for each in hidden])
        embeds = self.fc1(x)
        lstm_out, hidden = self.lstm(embeds, hidden)     
        # stack up lstm outputs
        #lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = out[:,t,:] #Extract hidden layer of the current time point considered
        
        out = F.log_softmax(out, dim=1)

        return out, hidden
    
    def predict_proba(self, x, hidden, t):
        #batch_size = x.size(0)
        #hidden = tuple([each.repeat(1, batch_size, 1).data for each in hidden])
        embeds = self.fc1(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
         # stack up lstm outputs
        #lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = out[:,t,:]
        
        out = F.softmax(out, dim=1)

        return out, hidden
    
    def get_belief(self, x, hidden):
        self.x = x
        embeds = self.fc1(x)
        ## aggregate the belief b
        self.b, hidden = self.lstm(embeds, hidden)
    
    def calculate_loss(self, t):
        ## sample a state at time t2 (see the reparametralization trick is used)
        z2_mu, z2_logsigma = self.b_to_z(self.b[:, (t+1), :])
        z2_epsilon = torch.randn_like(z2_mu)
        z2 = z2_mu + torch.exp(z2_logsigma)*z2_epsilon

        ## sample a state at time t1
        ## then infer state at time t1 based on states at time t2
        qs_z1_mu, qs_z1_logsigma = self.bz2_infer_z1(
            torch.cat((self.b[:,t,:], z2), dim = -1))
        qs_z1_epsilon = torch.randn_like(qs_z1_mu)
        qs_z1 = qs_z1_mu + torch.exp(qs_z1_logsigma)*qs_z1_epsilon

        #### After sampling states z from the variational distribution, we can calculate
        #### the loss.

        ## state distribution at time t1 based on belief at time 1
        pb_z1_mu, pb_z1_logsigma = self.b_to_z(self.b[:, t, :])

        ## state distribution at time t2 based on states at time t1 and state transition
        t_z2_mu, t_z2_logsigma = self.z1_to_z2(qs_z1)
        
        ## observation distribution at time t2 based on state at time t2
        x2 = self.z_to_x(z2)

        #### start calculating the loss

        #### KL divergence between z distribution at time t1 based on variational distribution
        #### (inference model) and z distribution at time t1 based on belief.
        #### This divergence is between two normal distributions and it can be calculated analytically
        
        ## KL divergence between pb_z1 and qs_z1
        loss = 0.5*torch.sum(((pb_z1_mu - qs_z1)/torch.exp(pb_z1_logsigma))**2,-1) + \
               torch.sum(pb_z1_logsigma, -1) - torch.sum(qs_z1_logsigma, -1)
        
        #### The following four terms estimate the KL divergence between the z distribution at time t2
        #### based on variational distribution (inference model) and z distribution at time t2 based on transition.
        #### In contrast with the above KL divergence for z distribution at time t1, this KL divergence
        #### can not be calculated analytically because the transition distribution depends on z_t1, which is sampled
        #### after z_t2. Therefore, the KL divergence is estimated using samples
        
        ## state log probabilty at time t2 based on belief
        loss += torch.sum(-0.5*z2_epsilon**2 - 0.5*z2_epsilon.new_tensor(2*np.pi) - z2_logsigma, dim = -1) 

        ## state log probabilty at time t2 based on transition
        loss += torch.sum(0.5*((z2 - t_z2_mu)/torch.exp(t_z2_logsigma))**2 + 0.5*z2.new_tensor(2*np.pi) + t_z2_logsigma, -1)
        
        ## reconstruction loss (observation at time t2 - prediction at time t2)
        loss += -torch.sum(self.x[:,t+1,:]*torch.log(x2) + (1-self.x[:,t+1,:])*torch.log(1-x2), -1)
        
        loss = torch.mean(loss)
        
        return loss
    
    def generate_next(self, x, hidden, t):
        self.get_belief(x, hidden)
        
        ## at time t, we sample a state z based on belief at time t
        z2_mu, z2_logsigma = self.b_to_z(self.b[:,t,:])
        z2_epsilon = torch.randn_like(z2_mu)
        z2 = z2_mu + torch.exp(z2_logsigma)*z2_epsilon 
        
        ## then infer state at time t1 based on states at time t2
        qs_z1_mu, qs_z1_logsigma = self.bz2_infer_z1(
            torch.cat((self.b[:,t-1,:], z2), dim = -1))
        qs_z1_epsilon = torch.randn_like(qs_z1_mu)
        qs_z1 = qs_z1_mu + torch.exp(qs_z1_logsigma)*qs_z1_epsilon       
       
        next_z_mu, next_z_logsigma = self.z1_to_z2(qs_z1)
        next_z_epsilon = torch.randn_like(next_z_mu)
        next_z = next_z_mu + torch.exp(next_z_logsigma)*next_z_epsilon
                        
        pred_x = self.z_to_x(next_z)
        
        return pred_x
    
    def generate_current(self, x, hidden, t):
        self.get_belief(x, hidden)
        
        ## at time t, we sample a state z based on belief at time t
        z_mu, z_logsigma = self.b_to_z(self.b[:,t,:])
        z_epsilon = torch.randn_like(z_mu)
        z = z_mu + torch.exp(z_logsigma)*z_epsilon 
                        
        pred_x = self.z_to_x(z)
        
        return pred_x
        
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
    
    def get_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).to(device),
                  weight.new(self.n_layers*2, batch_size, self.hidden_dim).to(device))
        return hidden

class DataLoaderCustom(Dataset):
    def __init__(self, features, labels=None, weights=None, transform=None):
        """
            Args:
                features (string): np array of features.
                transform (callable, optional): Optional transform to be applied
                    on a sample.
        """
        self.features = features
        self.labels = labels
        self.weights = weights
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {}
        sample['x'] = self.features[idx].astype('float32')
        if self.labels is not None:
            sample['y'] = self.labels[idx]
            if self.weights is not None:
                sample['w'] = self.weights[self.labels[idx]].astype('float32')
        return sample


def _check_targets(y_true, y_pred):
    check_consistent_length(y_true, y_pred)
    type_true = type_of_target(y_true)
    type_pred = type_of_target(y_pred)

    y_type = {type_true, type_pred}
    if y_type == {"binary", "multiclass"}:
        y_type = {"multiclass"}

    if len(y_type) > 1:
        raise ValueError("Classification metrics can't handle a mix of {0} "
                         "and {1} targets".format(type_true, type_pred))

    # We can't have more than one value on y_type => The set is no more needed
    y_type = y_type.pop()

    # No metrics support "multiclass-multioutput" format
    if (y_type not in ["binary", "multiclass", "multilabel-indicator"]):
        raise ValueError("{0} is not supported".format(y_type))

    if y_type in ["binary", "multiclass"]:
        y_true = column_or_1d(y_true)
        y_pred = column_or_1d(y_pred)
        if y_type == "binary":
            unique_values = np.union1d(y_true, y_pred)
            if len(unique_values) > 2:
                y_type = "multiclass"

    if y_type.startswith('multilabel'):
        y_true = csr_matrix(y_true)
        y_pred = csr_matrix(y_pred)
        y_type = 'multilabel-indicator'

    return y_type, y_true, y_pred


def normalize(cm, normalize=None, epsilon=1e-8):
    with np.errstate(all='ignore'):
        if normalize == 'true':
            cm = cm / (cm.sum(axis=1, keepdims=True) + epsilon)
        elif normalize == 'pred':
            cm = cm / (cm.sum(axis=0, keepdims=True) + epsilon)
        elif normalize == 'all':
            cm = cm / (cm.sum() + epsilon)
        cm = np.nan_to_num(cm)
    return cm


class ConfusionMatrixPlot:

    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels.copy()
        self.displabelsx = display_labels.copy()
        #if "Novel" in display_labels:
        #	self.displabelsx.remove("Novel")
        self.displabelsy = display_labels.copy()
        #self.displabelsy.remove("Unassigned")

    def plot(self, include_values=True, cmap='viridis',
             xticks_rotation='vertical', values_format=None, ax=None, fontsize=13):

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = '.2g'

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0
            for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(j, i,
                                           format(cm[i, j], values_format),
                                           ha="center", va="center",
                                           color=color, fontsize=fontsize)

        fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               )
        ax.set_xticklabels(self.displabelsx[:cm.shape[1]], fontsize=fontsize)
        ax.set_yticklabels(self.displabelsy[:cm.shape[0]], fontsize=fontsize)
        ax.set_xlabel(xlabel="Predicted label", fontsize=fontsize+2)
        ax.set_ylabel(ylabel="True label", fontsize=fontsize+2)

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self

def compute_ap(gts, preds):
    aps = []
    for i in range(preds.shape[1]):
        ap, prec, rec = calc_pr(gts == i, preds[:,i:i+1])
        aps.append(ap)
    aps = np.array(aps)
    return np.nan_to_num(aps)

def calc_pr(gt, out, wt=None):
    gt = gt.astype(np.float64).reshape((-1,1))
    out = out.astype(np.float64).reshape((-1,1))

    tog = np.concatenate([gt, out], axis=1)*1.
    ind = np.argsort(tog[:,1], axis=0)[::-1]
    tog = tog[ind,:]
    cumsumsortgt = np.cumsum(tog[:,0])
    cumsumsortwt = np.cumsum(tog[:,0]-tog[:,0]+1)
    prec = cumsumsortgt / (cumsumsortwt + 1e-8)
    rec = cumsumsortgt / (np.sum(tog[:,0]) + 1e-8)
    ap = voc_ap(rec, prec)
    return ap, rec, prec

def voc_ap(rec, prec):
    rec = rec.reshape((-1,1))
    prec = prec.reshape((-1,1))
    z = np.zeros((1,1)) 
    o = np.ones((1,1))
    mrec = np.vstack((z, rec, o))
    mpre = np.vstack((z, prec, z))

    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    I = np.where(mrec[1:] != mrec[0:-1])[0]+1;
    ap = np.sum((mrec[I] - mrec[I-1])*mpre[I])
    return ap

## Match function as in R
match = lambda a, b: [ b.index(x) if x in b else None for x in a ]