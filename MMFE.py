import os
import joblib
import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from FeatureExtraction import MolEncoder
from Embedding import MLP, Trainer, normalize, denormalize, get_metrics, get_metrics_classification


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Configs:
    def __init__(self, feature_name, trainer, scaler, get_features, output_dir):
        self.feature_name = feature_name
        self.trainer = trainer
        self.scaler = scaler
        self.get_features = get_features
        self.filepath_scaler = os.path.join(output_dir, f'scaler_{self.feature_name}.pkl')
        self.filepath_ckpt = os.path.join(output_dir, f'net_{self.feature_name}.pth')
        self.filepath_hist = os.path.join(output_dir, f'history_{self.feature_name}.csv')
        self.filepath_loss = os.path.join(output_dir, f'loss_{self.feature_name}.png')


class MMFE:
    def __init__(self, output_dir, device, name=0):
        self.output_dir = output_dir
        self.device = device
        self.name = f'{name}'
        
        ## neural networks
        self.encoder = MolEncoder(device)
        self.mlp_fpt = Trainer(MLP(self.encoder.feature_dim_fp), self.device)
        self.mlp_img = Trainer(MLP(self.encoder.feature_dim_img), self.device)
        self.mlp_smi = Trainer(MLP(self.encoder.feature_dim_smi), self.device)
        
        ## scaler
        self.scaler_fpt = MinMaxScaler()
        self.scaler_img = MinMaxScaler()
        self.scaler_smi = MinMaxScaler()
        
        ## weights
        weights = self.calc_weights()
        self.w_fpt = weights[0]
        self.w_img = weights[1]
        self.w_smi = weights[2]
        
        ## save
        with open(os.path.join(self.output_dir, f'weights_{self.name}.csv'), 'w') as f:
            f.write(f'w_fpt,{self.w_fpt}\n')
            f.write(f'w_img,{self.w_img}\n')
            f.write(f'w_smi,{self.w_smi}\n')
        
        
    def fit(self, X_tr, X_va, y_tr, y_va, temperature=1.0):
        ## fingerprint
        configs_fpt = Configs(f'fpt_{self.name}', self.mlp_fpt, self.scaler_fpt, self.encoder.get_fpt_features, self.output_dir)
        loss_fpt = _train(X_tr, X_va, y_tr, y_va, configs_fpt)
        
        ## molecular image
        configs_img = Configs(f'img_{self.name}', self.mlp_img, self.scaler_img, self.encoder.get_img_features, self.output_dir)
        loss_img = _train(X_tr, X_va, y_tr, y_va, configs_img)
        
        ## SMILES string
        configs_smi = Configs(f'smi_{self.name}', self.mlp_smi, self.scaler_smi, self.encoder.get_smi_features, self.output_dir)
        loss_smi = _train(X_tr, X_va, y_tr, y_va, configs_smi)
        
        return loss_fpt, loss_img, loss_smi
        
        
    def predict(self, X, weights=None):
        ## fingerprint
        inps_fpt = np.array([self.encoder.get_fpt_features(smi) for smi in X], dtype=np.float32)
        inps_fpt = self.scaler_fpt.transform(inps_fpt)
        embs_fpt = self.mlp_fpt.get_embeddings(inps_fpt)

        ## molecular image
        inps_img = np.array([self.encoder.get_img_features(smi) for smi in X], dtype=np.float32)
        inps_img = self.scaler_img.transform(inps_img)
        embs_img = self.mlp_img.get_embeddings(inps_img)

        ## SMILES string
        inps_smi = np.array([self.encoder.get_smi_features(smi) for smi in X], dtype=np.float32)
        inps_smi = self.scaler_smi.transform(inps_smi)
        embs_smi = self.mlp_smi.get_embeddings(inps_smi)

        ## fusion
        if weights is None:
            embs = self.w_fpt * embs_fpt + self.w_img * embs_img + self.w_smi * embs_smi
        else:
            embs = weights[0] * embs_fpt + weights[1] * embs_img + weights[2] * embs_smi

        return embs
        
        
    def load_model(self):
        ## fingerprint
        self.mlp_fpt.load_model(os.path.join(self.output_dir, f'net_fpt_{self.name}.pth'))
        self.scaler_fpt = joblib.load(os.path.join(self.output_dir, f'scaler_fpt_{self.name}.pkl'))

        ## molecular image
        self.mlp_img.load_model(os.path.join(self.output_dir, f'net_img_{self.name}.pth'))
        self.scaler_img = joblib.load(os.path.join(self.output_dir, f'scaler_img_{self.name}.pkl'))

        ## SMILES string
        self.mlp_smi.load_model(os.path.join(self.output_dir, f'net_smi_{self.name}.pth'))
        self.scaler_smi = joblib.load(os.path.join(self.output_dir, f'scaler_smi_{self.name}.pkl'))

        ## weights
        with open(os.path.join(self.output_dir, f'weights_{self.name}.csv')) as f:
            lines = f.readlines()
        self.w_fpt = float(lines[0].rstrip().split(',')[-1])
        self.w_img = float(lines[1].rstrip().split(',')[-1])
        self.w_smi = float(lines[2].rstrip().split(',')[-1])

    
    def calc_weights(self, alpha=0.2, beta=5.5):
        scores = np.array([
            sigmoid(alpha * np.sqrt(self.encoder.feature_dim_fp) - beta),
            sigmoid(alpha * np.sqrt(self.encoder.feature_dim_img) - beta),
            sigmoid(alpha * np.sqrt(self.encoder.feature_dim_smi) - beta),
        ])
        weights = scores / scores.sum()
        return weights
        

def _train(X_tr, X_va, y_tr, y_va, configs):
    ## preprocess
    inps_tr = np.array([configs.get_features(smi) for smi in X_tr], dtype=np.float32)
    inps_va = np.array([configs.get_features(smi) for smi in X_va], dtype=np.float32)

    ## scaler
    configs.scaler = configs.scaler.fit(inps_tr)
    inps_tr = configs.scaler.transform(inps_tr)
    inps_va = configs.scaler.transform(inps_va)
    joblib.dump(configs.scaler, configs.filepath_scaler)
    
    ## fit
    history = configs.trainer.fit(
        inps_tr, y_tr,
        num_epochs=2000, learning_rate=1e-4,
        validation_data={'X':inps_va, 'y':y_va},
        filepath_ckpt=configs.filepath_ckpt
    )

    ## history
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(configs.filepath_hist, index=False)

    ## loss plot
    fig, ax = plt.subplots(1,1,figsize=(8,4.5))
    ax = sns.lineplot(data=df_hist, x='epoch', y='loss', err_style=None, ax=ax)
    ax = sns.lineplot(data=df_hist, x='epoch', y='val_rmse', err_style=None, ax=ax)
    _ = ax.axvline(x=df_hist['val_rmse'].argmin(), c='r', linestyle='--')
    ax.legend(['RMSE(tr)', 'RMSE(va)'])
    plt.savefig(configs.filepath_loss)
    plt.close()

    ## best score (validation)
    loss_va = np.min(history['val_rmse'])
    
    ## restore the best checkpoint
    configs.trainer.load_model(configs.filepath_ckpt)
    
    return loss_va
    
    