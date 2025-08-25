import os
from .MMFE import *

## MolScribe
dirname_ckpt = os.path.join(os.path.dirname(__file__), "molscribe", "ckpts")
filepath_ckpt = os.path.join(dirname_ckpt, "swin_base_char_aux_1m680k.pth")

if not os.path.exists(dirname_ckpt):
    os.mkdir(dirname_ckpt)

if not os.path.exists(filepath_ckpt):
    os.system(f"wget https://huggingface.co/yujieq/MolScribe/resolve/main/swin_base_char_aux_1m680k.pth?download=true -o {filepath_ckpt}")

