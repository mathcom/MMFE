import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

def get_metrics(y_real, y_pred):
    metrics = {
        'rmse':np.sqrt(mean_squared_error(y_real, y_pred)),
        'r2':r2_score(y_real, y_pred),
        'pcc':pearsonr(y_real, y_pred)[0]
    }
    return metrics

def normalize(y):
    y = np.maximum(0.0001, np.minimum(0.9999, y))
    z = np.log(y) - np.log(1-y)
    return z


def denormalize(z):
    y = 1 / (1 + np.exp(-z))
    return y


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out
        

class MyDataset(Dataset):
    def __init__(self, data, labels, doNormalization=True):
        self.data = data
        self.labels = normalize(labels) if doNormalization else labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
        

class Interface:
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(device)
        
    def get_embeddings(self, x):
        x = torch.tensor(x, device=self.device)
        with torch.no_grad():
            out = self.model.relu(self.model.fc1(x))
        return out.cpu().numpy()
        
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        
    def load_model(self, filepath):
        weights = torch.load(filepath) #, weights_only=True)
        self.model.load_state_dict(weights)


class Trainer(Interface):
    def __init__(self, model, device):
        super(Trainer, self).__init__(model, device)
        
        
    def fit(self, X, y, batch_size=32, num_epochs=30, learning_rate=1e-3, validation_data=None, filepath_ckpt='model_state.pth'):
        '''
        X.shape : (total, input_dim)
        y.shape : (total, 1)
        '''
        ## dataset
        dataset = MyDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        ## mode: train
        self.model.train()
        
        ## initialize optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
                
        ## train
        self.history = {'epoch':[], 'loss':[], 'val_rmse':[], 'val_r2':[], 'val_pcc':[]}
        best_score = 1e+6
        for i in range(num_epochs):
            loss_avg = 0.
            for batch_idx, (inputs, targets) in enumerate(loader):
                ## device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                ## forward
                out = self.model(inputs)
                loss = F.mse_loss(out, targets)
                loss_avg += loss.item()
                
                ## backprogation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            ## logging
            loss_avg /= len(loader)
            self.history['epoch'].append(i)
            self.history['loss'].append(loss_avg)
            
            ## validation
            if validation_data:
                metrics = self.evaluate(validation_data['X'], validation_data['y'], batch_size)
                self.history['val_rmse'].append(metrics['rmse'])
                self.history['val_r2'].append(metrics['r2'])
                self.history['val_pcc'].append(metrics['pcc'])
                if metrics['rmse'] < best_score:
                    best_score = metrics['rmse']
                    ## save
                    self.save_model(filepath_ckpt)
            else:
                self.history['val_rmse'].append(0)
                self.history['val_r2'].append(0)
                self.history['val_pcc'].append(0)
                self.save_model(filepath_ckpt)
        
        self.model.eval()
        return self.history


    def evaluate(self, X, y, batch_size=32):
        '''
        X.shape : (total, input_dim)
        y.shape : (total, 1)
        '''
        ## dataset
        dataset = MyDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        ## mode: eval
        self.model.eval()
        
        ## predict
        y_real = []
        y_pred = []
        for batch_idx, (inputs, targets) in enumerate(loader):
            ## device
            inputs = inputs.to(self.device)
            
            ## forward
            with torch.no_grad():
                out = self.model(inputs)
            
            ## save
            y_real.append(targets.cpu().numpy())
            y_pred.append(out.cpu().numpy())
        
        ## stack & ravel
        y_real = np.vstack(y_real).ravel()
        y_pred = np.vstack(y_pred).ravel()
        
        ## metrics        
        return get_metrics(y_real, y_pred)
        
    
    def inference(self, x):
        x = torch.tensor(x, device=self.device)
        with torch.no_grad():
            out = self.model(x)
        return out.cpu().numpy()
        
        

