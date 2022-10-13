import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import tqdm

from model import SemanticSegmentation
from utils import log_train_info

num_classes = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

it = 0
num_per_log = 100

def train_seg(rgb, sem, model, optim): 
    rgb = rgb.float().permute(0,3,1,2).to(device) 
    sem = sem.long().to(device) 

    pred_sem = model(rgb)

    loss = F.cross_entropy(pred_sem, sem)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if it % num_per_log == 0:
        seg_info = dict(
            loss = float(loss), 
            rgb = rgb[0].permute(1,2,0).byte().cpu().detach().numpy(),
            sem = sem[0].cpu().detach().numpy(),
            pred_sem = pred_sem[0].cpu().detach().numpy().argmax(0)
        )
        log_train_info(seg_info)


    del rgb, sem, pred_sem, loss


def main(args):
    torch.manual_seed(args.seed)

    seg_model = SemanticSegmentation(num_classes)
    seg_optim = optim.Adam(seg_model.parameters(), lr=args.lr)

    batch_size = 64
    
    dataset = datasets.Cityscapes('./data/cityscapes', split='train', mode='fine',
                    target_type='semantic') 
    dataloader = DataLoader(dataset, batch_size)

    global it
    for epoch in range(args.num_epoch):
        for rgb, sem in tqdm.tqdm(dataloader, desc=f'Epoch {epoch}'):
            train_seg(rgb, sem, seg_model, seg_optim)            
            it += 1

    # Save model
    save_dir = args.save_dir
    seg_path = f'{save_dir}/seg_model.th'

    torch.save(seg_model.state_dict('seg'), seg_path)
    print (f'saved to {seg_path}')

 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', default='config.yaml')

    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

    # Training misc
    parser.add_argument('--num-epoch', type=int, default=1)
    parser.add_argument('--num-per-log', type=int, default=100, help='log per iter')
    parser.add_argument('--num-per-save', type=int, default=1, help='save per epoch')
    
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-workers', type=int, default=16)
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=2021)

    args = parser.parse_args()

    main(args)