import glob
import os
import os.path as osp
import argparse
import shutil
import numpy as np
import torch
import cv2
import RRDBNet_arch as arch
from huggingface_hub import hf_hub_download

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True, default="cpu",
    choices=["cpu", "cuda"], type=str,
    help="device to use for upscaling (cpu or cuda (nvidia gpu))")
args = vars(ap.parse_args())

test_img_folder = 'LR/*'
curr = 'LR/'
old = 'old/'

os.makedirs('results', exist_ok=True)
os.makedirs('weights', exist_ok=True)
os.makedirs(old, exist_ok=True)

requested_device = args["device"]
if requested_device == 'cuda' and not torch.cuda.is_available():
    print('Warning: CUDA not available, falling back to CPU.')
    requested_device = 'cpu'
device = torch.device(requested_device)

# Download model weights from Hugging Face if not already present
model_path = 'weights/RealESRGAN_x4.pth'
if not os.path.exists(model_path):
    print('Downloading model weights from Hugging Face...')
    hf_hub_download(
        repo_id='ai-forever/Real-ESRGAN',
        filename='RealESRGAN_x4.pth',
        local_dir='weights'
    )
    print('Download complete.')

def remap_keys(state_dict):
    """Remap ai-forever/Real-ESRGAN key names to match RRDBNet_arch.py."""
    new_sd = {}
    for k, v in state_dict.items():
        k = k.replace('conv_body', 'trunk_conv')  # must be before 'body.' replacement
        k = k.replace('body.', 'RRDB_trunk.')
        k = k.replace('.rdb1.', '.RDB1.')
        k = k.replace('.rdb2.', '.RDB2.')
        k = k.replace('.rdb3.', '.RDB3.')
        k = k.replace('conv_up1', 'upconv1')
        k = k.replace('conv_up2', 'upconv2')
        k = k.replace('conv_hr', 'HRconv')
        new_sd[k] = v
    return new_sd

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(remap_keys(state_dict), strict=True)
model.eval()
model = model.to(device)

print('Model loaded. Running inference...')

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(f'{idx}: {base}')

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(f'results/{base}_rlt.png', output)

print(f'Done. {idx} image(s) upscaled to results/')

# Move processed input images to old/
for file_name in os.listdir(curr):
    source_path = os.path.join(curr, file_name)
    destination_path = os.path.join(old, file_name)
    shutil.move(source_path, destination_path)
    print(f"Moved {file_name} to {old}")
