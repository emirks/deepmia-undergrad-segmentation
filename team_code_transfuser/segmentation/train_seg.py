import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

import tqdm

# from models.ERFNet.model import SemanticSegmentation as SegmentationModel
# from models.PIDNetLidar.model import PIDNet as SegmentationModel
from models.PIDNetLidarv2.model import SegmentationModel
# from models.PIDNetLidarSAM.model import SegmentationModel
# from models.PIDNet.model import PIDNet as SegmentationModel
# from models.TransfuserModel.model import SegmentationModel

from seg_dataset_carla import SegmentationDataset as SegmentationDatasetCarla
from seg_dataset_a2d2 import SegmentationDataset as SegmentationDatasetA2d2

from utils import visualize_semantic_processed, smooth_loss, BondaryLoss, adjust_learning_rate, get_confusion_matrix, visualize_cm, calculate_IoU_from_cm
import config

import os
from matplotlib import pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

sem_loss = nn.CrossEntropyLoss(weight=config.class_weights)
bd_loss = BondaryLoss()

def train_batch(batch, model : SegmentationModel, optim, epoch): 
    rgb, sem, pred_sem, loss = SegmentationModel.pass_from_model(model, batch, sem_loss, smooth_loss, bd_loss, device)

    adjust_learning_rate(optim, config.base_lr, total_iterations, train_it)
    writer.add_scalar("Loss/train-per-batch", loss, train_it)

    optim.zero_grad()
    loss.backward()
    optim.step()

    del rgb, sem, pred_sem
    return loss

def val_batch(batch, model : SegmentationModel): 
    rgb, sem, pred_sem, loss = SegmentationModel.pass_from_model(model, batch, sem_loss, smooth_loss, bd_loss, device)
    sem = sem.cpu().detach().numpy()
    pred_sem = pred_sem.cpu().detach().numpy().argmax(1)
    confusion_mat = get_confusion_matrix(sem, pred_sem)

    writer.add_scalar("Loss/val-per-batch", loss, val_it)

    if val_it % args.num_per_log == 0: 
        loss = float(loss) 
        rgb_vis = rgb[0].permute(1,2,0).byte().cpu().detach().numpy()
        sem_vis = sem[0]
        pred_sem_vis = pred_sem[0]
        f, [ax1, ax2, ax3] = plt.subplots(1,3,figsize=(32, 10))
        f.text(.01, .99, f"loss: {loss}", size = 20, ha='left', va='top')
        ax1.imshow(rgb_vis)
        ax2.imshow(visualize_semantic_processed(sem_vis))
        ax3.imshow(visualize_semantic_processed(pred_sem_vis))
        plt.savefig(f"{config.SAVE_DIR}/carla/logs/{args.model_name}/log-{val_it // args.num_per_log}.png")
        del rgb_vis, sem_vis, pred_sem_vis

    del rgb, sem, pred_sem
    return loss, confusion_mat

def train_model(model, dataloader, optim, epoch): 
    model.train()
    total_loss = 0

    global train_it
    for batch in tqdm.tqdm(dataloader, desc=f'Epoch {epoch}-training'):
        loss = train_batch(batch, model, optim, epoch)
        total_loss += loss
        train_it += 1
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("Avg-loss/train-per-epoch", avg_loss, epoch)


def val_model(model, dataloader, epoch): 
    model.eval()
    total_loss = 0
    confusion_matrix = np.zeros((len(config.labels) + 1, len(config.labels) + 1))

    global val_it
    with torch.no_grad():    
        for batch in tqdm.tqdm(dataloader, desc=f'Epoch {epoch}-validating'):
            loss, confusion_mat = val_batch(batch, model)
            total_loss += loss
            confusion_matrix += confusion_mat
            val_it += 1
    avg_loss = total_loss / len(dataloader)    

    mean_IoU = calculate_IoU_from_cm(confusion_matrix)
    writer.add_scalar("Avg-loss/val-per-epoch", avg_loss, epoch)
    writer.add_figure("Confusion matrix", visualize_cm(confusion_matrix, config.labels), epoch)
    writer.add_scalar("Mean IoU/val-per-epoch", mean_IoU, epoch)
    return avg_loss, mean_IoU


def main(args):
    torch.manual_seed(args.seed)

    seg_model = SegmentationModel.initialize(len(config.labels) + 1)
    seg_model.to(device)
    seg_optim = optim.SGD(seg_model.parameters(), 
                            lr=config.base_lr,
                            momentum=config.optim_momentum,
                            weight_decay=config.optim_wd)
    
    full_dataset = SegmentationDatasetA2d2() 
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    # train_datasets = [SegmentationDatasetCarla(hdf5_file_name=town_name) for town_name in config.towns]
    # train_dataset = ConcatDataset(datasets=train_datasets)
    # val_datasets = [SegmentationDatasetCarla(hdf5_file_name=f"{town_name}-val") for town_name in config.towns]
    # val_dataset = ConcatDataset(datasets=val_datasets)

    train_dataloader = DataLoader(train_dataset, 
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(val_dataset, 
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    epoch_iters = int(train_dataset.__len__() / args.batch_size)
    global total_iterations
    total_iterations = args.num_epoch * epoch_iters

    log_dir = f"{config.SAVE_DIR}/carla/logs/{args.model_name}"
    if not os.path.exists(log_dir): 
        os.makedirs(log_dir)

    model_save_path = f'{config.SAVE_DIR}/carla/pretrained/{args.model_name}'
    best_mIoU = 0
    try:
        for epoch in range(args.num_epoch): 
            train_model(seg_model, train_dataloader, seg_optim, epoch)

            ave_loss, mean_IoU = val_model(seg_model, val_dataloader, epoch)
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(ave_loss, mean_IoU, best_mIoU)
            print(msg)
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(seg_model.state_dict(), f"{model_save_path}-best.pt")
                print(f'Model with best mIoU saved to {model_save_path}-best.pt.')
        writer.flush()
        writer.close()                 
    except KeyboardInterrupt:
        # if someone hits ctrl+c, save it before exiting
        print("Saving incomplete model.") 


    torch.save(seg_model.state_dict(), f"{model_save_path}.pt")
    print(f'Saved to {model_save_path}.th')

 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', default='config.yaml')
    parser.add_argument('--model-name', default='model')

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

    # Global variables
    train_it = 0
    val_it = 0
    total_iterations = 0
    # tensorboard writer
    writer = SummaryWriter(f"runs/{args.model_name}")
    
    main(args)