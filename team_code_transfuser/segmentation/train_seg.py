import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize
import tqdm

#from models.ERFNet.model import SemanticSegmentation
from models.PIDNet.model import get_pred_model
from seg_dataset import SegmentationDataset
from utils import log_train_info, labels

num_classes = 5
seg_channels = labels
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

it = 0

def train_seg(rgb, sem, model, optim, visualize_log): 
    rgb = rgb.float().permute(0,3,1,2).to(device) 
    sem = sem.long().to(device) 

    pred_sem = model(rgb)
    resize = Resize(size = (sem.shape[1], sem.shape[2]))
    pred_sem = resize(pred_sem)

    loss = F.cross_entropy(pred_sem, sem)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if visualize_log and it % args.num_per_log == 0:
        seg_info = dict(
            loss = float(loss), 
            rgb = rgb[0].permute(1,2,0).byte().cpu().detach().numpy(),
            sem = sem[0].cpu().detach().numpy(),
            pred_sem = pred_sem[0].cpu().detach().numpy().argmax(0)
        )
        log_train_info(seg_info, it // args.num_per_log)


    del rgb, sem, pred_sem, loss


def main(args):
    torch.manual_seed(args.seed)

    #seg_model = SemanticSegmentation(len(seg_channels) + 1).to(device)
    seg_model = get_pred_model("PIDNet-m", len(seg_channels) + 1).to(device)
    seg_optim = optim.Adam(seg_model.parameters(), lr=args.lr)
    
    dataset = SegmentationDataset()
    dataloader = DataLoader(dataset, 
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    global it
    for epoch in range(args.num_epoch):
        for rgb, sem in tqdm.tqdm(dataloader, desc=f'Epoch {epoch}'):
            train_seg(rgb, sem, seg_model, seg_optim, args.visualize)            
            it += 1

    # Save model
    seg_path = f'{args.save_path}/seg_model2.th'

    torch.save(seg_model.state_dict(), seg_path)
    print (f'saved to {seg_path}')

 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', default='config.yaml')
    parser.add_argument('--save_path', default='.')

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