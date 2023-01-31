import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data._utils import collate

import cv2

from models.PIDNet.model import get_pred_model
import config
from utils import log_eval_info, load_pretrained
from seg_dataset import SegmentationDataset


device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    torch.manual_seed(args.seed)

    seg_model = get_pred_model("PIDNet-m", len(config.labels) + 1).to(device)

    # Load model
    load_path = f'./pretrained/{args.load_path}.th'
    seg_model = load_pretrained(seg_model, load_path)
    seg_model.eval()
    print ("Model and weights loaded successfully")

    with torch.no_grad(): 
        input_img = cv2.imread("assets/rgb_0.png", cv2.IMREAD_COLOR)
        input_img = collate.default_collate(input_img).unsqueeze(0)
        input_img = input_img.float().permute(0,3,1,2).to(device)
        pred_sem = seg_model(input_img)
        resize = transforms.Resize(size = (input_img.shape[2], input_img.shape[3]))
        pred_sem = resize(pred_sem)

        seg_info = dict(
            rgb = input_img[0].permute(1,2,0).byte().cpu().detach().numpy(),
            pred_sem = pred_sem[0].cpu().detach().numpy().argmax(0)
        )
        log_eval_info(seg_info)

if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--load-path')
    # Reproducibility
    parser.add_argument('--seed', type=int, default=2021)

    args = parser.parse_args()

    main(args)