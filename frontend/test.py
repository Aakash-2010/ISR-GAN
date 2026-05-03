import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import argparse
import shutil
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True, default="cpu",
	choices=["cpu", "cuda"], type=str,
	help="device to use for training (cpu or cuda (nvidia gpu))")
args = vars(ap.parse_args())

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
if args["device"] == "cpu":
    device = torch.device('cpu')

elif args["device"] == "cuda":
    device = torch.device('cuda')

test_img_folder = 'LR/*'
curr = 'LR/'
old = 'old/'


model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)


# Listing all files in the test_img_folder
files_to_move = os.listdir(curr)

for file_name in files_to_move:
    source_path = os.path.join(curr, file_name)
    destination_path = os.path.join(old, file_name)
    
    # Move the file from source to destination
    shutil.move(source_path, destination_path)
    print(f"Moved {file_name} to {old}")
