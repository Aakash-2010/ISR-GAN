import gradio as gr
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from RRDBNet_arch import RRDBNet

MODEL_REPO = "ai-forever/Real-ESRGAN"
MODEL_FILE = "RealESRGAN_x4.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remap_keys(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        k = k.replace('conv_body', 'trunk_conv')
        k = k.replace('body.', 'RRDB_trunk.')
        k = k.replace('.rdb1.', '.RDB1.')
        k = k.replace('.rdb2.', '.RDB2.')
        k = k.replace('.rdb3.', '.RDB3.')
        k = k.replace('conv_up1', 'upconv1')
        k = k.replace('conv_up2', 'upconv2')
        k = k.replace('conv_hr', 'HRconv')
        k = k.replace('conv_last', 'conv_last')
        new_sd[k] = v
    return new_sd


def load_model():
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(remap_keys(state_dict), strict=True)
    model.eval()
    return model.to(DEVICE)


model = load_model()


def upscale(image: Image.Image):
    if image is None:
        return None
    image = image.convert("RGB")
    w, h = image.size
    if w * h > 800 * 800:
        image.thumbnail((800, 800), Image.LANCZOS)

    img_np = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(
        np.transpose(img_np[:, :, [2, 1, 0]], (2, 0, 1))
    ).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(tensor).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    out = np.transpose(out[[2, 1, 0], :, :], (1, 2, 0))
    return Image.fromarray((out * 255.0).round().astype(np.uint8))


demo = gr.Interface(
    fn=upscale,
    inputs=gr.Image(type="pil", label="Low-resolution input"),
    outputs=gr.Image(type="pil", label="4× upscaled output"),
    title="ESRGAN Super Resolution",
    description="Upload a low-resolution image to upscale it 4× using Real-ESRGAN.",
    allow_flagging="never",
)

demo.launch()
