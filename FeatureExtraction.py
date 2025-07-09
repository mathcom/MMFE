import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from rdkit.Chem import AllChem, Draw, MolFromSmiles, MolToSmiles
from transformers import AutoModelForMaskedLM, AutoTokenizer
from molscribe.model import Encoder


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Args:
    def __init__(self):
        self.encoder = 'swin_base_patch4_window12_384'
        self.use_checkpoint = False


class MolEncoder:
    def __init__(self, device=None):
        ## torch device
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        ## fp - Morgan
        self.feature_dim_fp = 2048
        self.morgan = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        
        ## txt - SMILES
        self.feature_dim_smi = 600
        self.chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM", use_safetensors=True)
        self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        #for param in self.chemberta.parameters():
        #    param.requires_grad = False
        
        ## img - MolScribe
        self.feature_dim_img = 1024
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.ocsr = self._load_pretrained_ocsr()
        #for param in self.ocsr.parameters():
        #    param.requires_grad = False
        
        
    def _load_pretrained_ocsr(self):
        encoder = Encoder(Args(), pretrained=False)
        
        ## load states
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'molscribe', 'ckpts', 'swin_base_char_aux_1m680k.pth')
        states = torch.load(filepath, map_location=self.device, weights_only=True)['encoder']
        
        ## safe load
        def _remove_prefix(state_dict):
            return {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        missing_keys, unexpected_keys = encoder.load_state_dict(_remove_prefix(states), strict=False)
        #if missing_keys:
        #    print('Missing keys: ' + str(missing_keys))
        #if unexpected_keys:
        #    print('Unexpected keys: ' + str(unexpected_keys))
        
        ## mode:eval
        encoder.eval()
        encoder.trainable = False
        return encoder.to(self.device)
        
        
    def get_fpt_features(self, smi):
        mol = MolFromSmiles(smi)
        if mol:
            fp = self.morgan.GetFingerprint(mol)
            return np.array(fp.ToList())
        else:
            return np.zeros(2048, dtype=int)
            
            
    def get_smi_features(self, smi, padding=True, kekuleSmiles=True, rootedAtAtom=0):
        if rootedAtAtom > 0:
            mol = MolFromSmiles(smi)
            rootedAtAtom = rootedAtAtom % mol.GetNumAtoms()
            smi = MolToSmiles(mol, kekuleSmiles=kekuleSmiles, rootedAtAtom=rootedAtAtom)
            
        with torch.no_grad():
            encoded_input = self.tokenizer(smi, return_tensors="pt", padding=padding, truncation=False)
            model_output = self.chemberta(**encoded_input)
            embedding_cls = model_output[0][::,0,::][0]
        return embedding_cls.numpy()
            

    def get_img_features(self, smi, rotation_angle=0):
        mol = MolFromSmiles(smi)
        if mol:
            img = Draw.MolToImage(mol)
            if rotation_angle % 360 > 0:
                img = self._rotate_image(img, rotation_angle)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature, _ = self.ocsr(img_tensor)
                feature_avg_pooled = torch.mean(feature, dim=1).squeeze()
            return feature_avg_pooled.cpu().numpy()
        else:
            return np.zeros(1024)
    

    def _rotate_image(self, img, angle):
        cv2_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        (h, w) = cv2_image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(cv2_image, M, (w, h))
        return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))