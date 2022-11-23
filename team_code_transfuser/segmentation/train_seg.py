import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize
import tqdm

from models.PIDNet.model import PIDNet
from seg_dataset import SegmentationDataset
from utils import log_train_info, visualize_semantic_processed, get_smooth_loss
import config

from matplotlib import pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

it = 0

def train_seg(rgb, sem, model, optim, visualize_log): 
    rgb = rgb.float().permute(0,3,1,2).to(device) 
    sem = sem.long().to(device) 

    pred_sem = model(rgb)
    out_p_loss, disparity, pred_sem, out_d_loss = model(rgb)
    resize = Resize(size = (rgb.shape[2], rgb.shape[3]))
    pred_sem = resize(pred_sem)

    loss = F.cross_entropy(pred_sem, sem)
    loss += out_p_loss
    loss += out_d_loss
    # calculate smoothness and add it to the loss
    mean_disp = disparity.mean(2, True).mean(3, True)
    norm_disp = disparity / (mean_disp + 1e-7)
    smooth_loss = get_smooth_loss(norm_disp, rgb)
    loss += config.disparity_smoothness * smooth_loss

    optim.zero_grad()
    loss.backward()
    optim.step()

    if visualize_log and it % args.num_per_log == 0:
        loss = float(loss) 
        rgb_vis = rgb[0].permute(1,2,0).byte().cpu().detach().numpy()
        sem_vis = sem[0].cpu().detach().numpy()
        pred_sem_vis = pred_sem[0].cpu().detach().numpy().argmax(0)
        f, [ax1, ax2, ax3] = plt.subplots(1,3,figsize=(32, 10))
        f.text(.01, .99, f"loss: {loss}", size = 20, ha='left', va='top')
        ax1.imshow(rgb_vis)
        ax2.imshow(visualize_semantic_processed(sem_vis))
        ax3.imshow(visualize_semantic_processed(pred_sem_vis))
        plt.savefig(f"./logs/log-{it // args.num_per_log}.png")

        #log_train_info(seg_info, it // args.num_per_log)


    del rgb, sem, pred_sem, loss


def main(args):
    torch.manual_seed(args.seed)

    # seg_model = get_pred_model("PIDNet-m", len(config.labels) + 1).to(device)
    seg_model = PIDNet(m=2, n=3, num_classes=len(config.labels) + 1, planes=64, ppm_planes=96, head_planes=128, augment=True)
    seg_optim = optim.Adam(seg_model.parameters(), lr=args.lr)
    
    datasets = [SegmentationDataset(hdf5_file_name=town_name) for town_name in config.towns]
    combined_dataset = ConcatDataset(datasets=datasets)
    dataloader = DataLoader(combined_dataset, 
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    global it
    try:
        for epoch in range(args.num_epoch): 
            for rgb, sem in tqdm.tqdm(dataloader, desc=f'Epoch {epoch}'):
                train_seg(rgb, sem, seg_model, seg_optim, args.visualize)            
                it += 1
    except KeyboardInterrupt:
        # if someone hits ctrl+c, save it before exiting
        print("Saving incomplete model.") 

    # Save model
    seg_path = f'./pretrained/{args.save_path}.th'

    torch.save(seg_model.state_dict(), seg_path)
    print (f'saved to {seg_path}')

 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', default='config.yaml')
    parser.add_argument('--save_path', default='model')

    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

    # Training misc
    parser.add_argument('--num-epoch', type=int, default=5)
    parser.add_argument('--num-per-log', type=int, default=100, help='log per iter')
    parser.add_argument('--num-per-save', type=int, default=1, help='save per epoch')
    
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('-visualize', '--visualize', action="store_true")
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=2021)

    args = parser.parse_args()

    main(args)