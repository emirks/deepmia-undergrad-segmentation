import torch
from torch import nn
from torchvision import transforms

from PIL import Image

from model import SemanticSegmentation
from utils import log_eval_info

class ERFNet(nn.Module):
    def __init__(self, seg_channels):

        super().__init__()

        self.erfnet = SemanticSegmentation(len(seg_channels)+1)

    def forward(self, rgb):

        return self.erfnet(rgb)

num_classes = 5
seg_channels = [4,6,7,10]
device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    torch.manual_seed(args.seed)

    seg_model = ERFNet(seg_channels).to(device)

    # Load model
    load_dir = "."
    load_path = f'{load_dir}/seg_1.th'

    seg_model.load_state_dict(torch.load(load_path))
    seg_model.eval()
    print ("Model and weights loaded successfully")

    # read the input and make it ready for the model
    #input_img = Image.open("assets/cityscapes-ex.png")
    input_img = Image.open("assets/carla-ex.jpeg")
    transform = transforms.Compose([
        transforms.Resize((168, 336), Image.BILINEAR), 
        transforms.ToTensor(),
    ])
    input_img = transform(input_img).unsqueeze(0).to(device)

    with torch.no_grad(): 
        output = seg_model(input_img)

    seg_info = dict(
        rgb = input_img[0].permute(1,2,0).cpu().detach().numpy(),
        pred_sem = output[0].max(0)[1].cpu().detach().numpy()
    )
    log_eval_info(seg_info)

    # label = output[0].max(0)[1].byte().cpu().data
    # label_colored = Colorize()(label.unsqueeze(0))

    # label_colored = transforms.ToPILImage()(label_colored)
    # label_colored.save("cityscapes-ex-label.png")


if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--load-dir')
    # Reproducibility
    parser.add_argument('--seed', type=int, default=2021)

    args = parser.parse_args()

    main(args)