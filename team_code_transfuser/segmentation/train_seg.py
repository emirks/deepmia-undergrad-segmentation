import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, CenterCrop
# from torchvision.transforms.functional import crop
import tqdm

# from models.ERFNet.model import SemanticSegmentation as ERFNet
from models.PIDNetLidar.model import PIDNet
from seg_dataset import SegmentationDataset
from utils import log_train_info, visualize_semantic_processed, get_smooth_loss, BondaryLoss, adjust_learning_rate, get_confusion_matrix
import config

from matplotlib import pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

it = 0
total_iterations = 0

sem_loss = nn.CrossEntropyLoss(weight=config.class_weights)
bd_loss = BondaryLoss()

# def pass_from_model(rgb, sem, edge, model): 
#     if config.split_cameras: 
#         camera_count = len(config.camera_rots)
#         rgbs = torch.tensor_split(rgb, camera_count, dim=3)
#         sems = torch.tensor_split(sem, camera_count, dim=2)
#         edges = torch.tensor_split(edge, camera_count, dim=2)
#         pred_sems = []
#         losses = []
#         for i in range(camera_count):
#             rgb_i, sem_i, edge_i = rgbs[i], sems[i], edges[i] 
#             out_p_loss, pred_sem, out_d_loss = model(rgb_i)
#             resize = Resize(size = (rgb_i.shape[2], rgb_i.shape[3]))
#             pred_sem = resize(pred_sem)
#             out_p_loss = resize(out_p_loss)
#             out_d_loss = resize(out_d_loss)

#             loss_s = sem_loss(pred_sem, sem_i)
#             loss_b = BondaryLoss()(out_d_loss, edge_i)
#             loss = loss_s + loss_b
            
#             # calculate smoothness and add it to the loss
#             disparity = nn.Sigmoid()(pred_sem)
#             mean_disp = disparity.mean(2, True).mean(3, True)
#             norm_disp = disparity / (mean_disp + 1e-7)
#             smooth_loss = get_smooth_loss(norm_disp, rgb_i)
#             loss += config.disparity_smoothness * smooth_loss     
#             loss = torch.unsqueeze(loss,0).mean()   
            
#             losses.append(loss)
#             pred_sems.append(pred_sem)
#         pred_sem = torch.cat(pred_sems, dim=3) 
#         loss = sum(losses)
#     else: 
#         out_p_loss, pred_sem, out_d_loss = model(rgb)
#         resize = Resize(size = (rgb.shape[2], rgb.shape[3]))
#         pred_sem = resize(pred_sem)
#         out_p_loss = resize(out_p_loss)
#         out_d_loss = resize(out_d_loss)
#         disparity = nn.Sigmoid()(pred_sem)

#         loss_s = sem_loss(pred_sem, sem)
#         loss_b = BondaryLoss()(out_d_loss, edge)

#         loss = loss_s + loss_b
#         # calculate smoothness and add it to the loss
#         mean_disp = disparity.mean(2, True).mean(3, True)
#         norm_disp = disparity / (mean_disp + 1e-7)
#         smooth_loss = get_smooth_loss(norm_disp, rgb)
#         loss += config.disparity_smoothness * smooth_loss
#         loss = torch.unsqueeze(loss,0).mean()

#     return pred_sem, loss

def pass_from_model(batch, model):
    rgb, lidar_bev, fused, sem, edge = batch
    rgb = rgb.float().permute(0,3,1,2).to(device) 
    lidar_bev = lidar_bev.float().permute(0,3,1,2).to(device)
    fused = fused.float().permute(0,3,1,2).to(device)
    sem = sem.long().to(device) 
    edge = edge.float().to(device)

    pred_sem = model(rgb, lidar_bev, fused)
    resize = Resize(size = (rgb.shape[2], rgb.shape[3]))
    pred_sem = resize(pred_sem)
    disparity = nn.Sigmoid()(pred_sem)
    loss = sem_loss(pred_sem, sem)

    # calculate smoothness and add it to the loss
    mean_disp = disparity.mean(2, True).mean(3, True)
    norm_disp = disparity / (mean_disp + 1e-7)
    smooth_loss = get_smooth_loss(norm_disp, rgb)
    loss += config.disparity_smoothness * smooth_loss
    loss = torch.unsqueeze(loss,0).mean()

    return rgb, sem, pred_sem, loss


def train_batch(batch, model, optim): 
    rgb, sem, pred_sem, loss = pass_from_model(batch, model)

    adjust_learning_rate(optim, config.base_lr, total_iterations, it)

    optim.zero_grad()
    loss.backward()
    optim.step()

    del rgb, sem, pred_sem, loss

def val_seg(batch, model): 
    rgb, sem, pred_sem, loss = pass_from_model(batch, model)
    sem = sem.cpu().detach().numpy()
    pred_sem = pred_sem.cpu().detach().numpy().argmax(1)
    confusion_mat = get_confusion_matrix(sem, pred_sem, config.labels)

    if it % args.num_per_log == 0: 
        loss = float(loss) 
        rgb_vis = rgb[0].permute(1,2,0).byte().cpu().detach().numpy()
        sem_vis = sem[0]
        pred_sem_vis = pred_sem[0]
        f, [ax1, ax2, ax3] = plt.subplots(1,3,figsize=(32, 10))
        f.text(.01, .99, f"loss: {loss}", size = 20, ha='left', va='top')
        ax1.imshow(rgb_vis)
        ax2.imshow(visualize_semantic_processed(sem_vis))
        ax3.imshow(visualize_semantic_processed(pred_sem_vis))
        plt.savefig(f"./logs/log-{it // args.num_per_log}.png")

    del rgb, sem, pred_sem
    return loss, confusion_mat

def train_model(model, dataloader, optim, epoch): 
    model.train()
    for batch in tqdm.tqdm(dataloader, desc=f'Epoch {epoch}-training'):
        train_batch(batch, model, optim)

def val_model(model, dataloader, epoch): 
    model.eval()
    total_loss = 0
    confusion_matrix = np.zeros((len(config.labels), len(config.labels)))

    global it
    with torch.no_grad():    
        for batch in tqdm.tqdm(dataloader, desc=f'Epoch {epoch}-validating'):
            loss, confusion_mat = val_seg(batch, model)
            total_loss += loss
            confusion_matrix += confusion_mat
            it += 1
    ave_loss = total_loss / len(dataloader)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    return ave_loss, mean_IoU


def main(args):
    torch.manual_seed(args.seed)

    # seg_model = get_pred_model("PIDNet-m", len(config.labels) + 1).to(device)
    seg_model = PIDNet(m=2, n=3, num_classes=len(config.labels) + 1, planes=64, ppm_planes=96, head_planes=128, augment=True)
    seg_model.to(device)
    seg_optim = optim.SGD(seg_model.parameters(), 
                            lr=config.base_lr,
                            momentum=config.optim_momentum,
                            weight_decay=config.optim_wd)
    
    train_datasets = [SegmentationDataset(hdf5_file_name=town_name) for town_name in config.towns]
    train_combined_dataset = ConcatDataset(datasets=train_datasets)
    train_dataloader = DataLoader(train_combined_dataset, 
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_datasets = [SegmentationDataset(hdf5_file_name=f"{town_name}-val") for town_name in config.towns]
    val_combined_dataset = ConcatDataset(datasets=val_datasets)
    val_dataloader = DataLoader(val_combined_dataset, 
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    epoch_iters = int(train_combined_dataset.__len__() / args.batch_size)
    global total_iterations
    total_iterations = args.num_epoch * epoch_iters

    seg_path = f'./pretrained/{args.save_path}.th'
    best_mIoU = 0
    try:
        for epoch in range(args.num_epoch): 
            train_model(seg_model, train_dataloader, seg_optim, epoch)

            ave_loss, mean_IoU = val_model(seg_model, val_dataloader, epoch)
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(ave_loss, mean_IoU, best_mIoU)
            print(msg)
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(seg_model.state_dict(), f"./pretrained/{args.save_path}-best.pt")
                print(f'Model with best mIoU saved to {args.save_path}-best.pt.')
            # torch.save(seg_model.state_dict(), seg_path)
                 
    except KeyboardInterrupt:
        # if someone hits ctrl+c, save it before exiting
        print("Saving incomplete model.") 


    torch.save(seg_model.state_dict(), seg_path)
    print(f'saved to {seg_path}')

 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', default='config.yaml')
    parser.add_argument('--save_path', default='model')

    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

    # Training misc
    parser.add_argument('--num-epoch', type=int, default=5)
    parser.add_argument('--num-per-log', type=int, default=10, help='log per iter')
    parser.add_argument('--num-per-save', type=int, default=1, help='save per epoch')
    
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('-visualize', '--visualize', action="store_true")
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=2021)

    args = parser.parse_args()

    main(args)