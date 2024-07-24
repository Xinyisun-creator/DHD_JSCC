import os
import numpy as np 
import torch
import torch.nn as nn
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
import json

from JSCC_get_args import get_args
from JSCC_modules import *
from dataset import CIFAR10, ImageNet, Kodak
from JSCC_utils import *
from JSCC_coop_network import *
from DHD_models import *
from DHD_loss import *
from DHD_Retrieval import DoRetrieval
from DHD_modifiedALEX import ModifiedAlexNet
from Dataloader import Loader
import matplotlib.pyplot as plt

#########
#   Parser for JSCC
#########

def get_args_parser():
    parser = argparse.ArgumentParser()

    ### Arguments for JSCC
    parser.add_argument('-JSCC_dataset', default='cifar')
    parser.add_argument('-JSCC_cout', type=int, default=12)
    parser.add_argument('-JSCC_cfeat', type=int, default=256)
    parser.add_argument('-JSCC_distribute', default=False)
    parser.add_argument('-JSCC_res', default=True)
    parser.add_argument('-JSCC_diversity', default=True)
    parser.add_argument('-JSCC_adapt', default=True)
    parser.add_argument('-JSCC_Nt', default=2)
    parser.add_argument('-JSCC_P1', default=10.0)
    parser.add_argument('-JSCC_P2', default=10.0)
    parser.add_argument('-JSCC_P1_rng', default=4.0)
    parser.add_argument('-JSCC_P2_rng', default=4.0)
    parser.add_argument('-JSCC_Nr', default=2)
    parser.add_argument('-JSCC_epoch', type=int, default=400)
    parser.add_argument('-JSCC_lr', type=float, default=1e-4)
    parser.add_argument('-JSCC_train_patience', type=int, default=12)
    parser.add_argument('-JSCC_train_batch_size', type=int, default=64)
    parser.add_argument('-JSCC_val_batch_size', type=int, default=32)
    parser.add_argument('-JSCC_resume', default=False)
    parser.add_argument('-JSCC_path', default='models/')

    args = parser.parse_args()

    return args

def save_tensor_as_image(tensor, filename):
    # Convert tensor to PIL image
    transform = transforms.ToPILImage()
    img = transform(tensor.cpu())
    
    # Save image
    img.save(filename)

#########
#           Parameter Setting
#########

dname = 'nuswide'
path = './datasets/'
args = get_args_parser()

Img_dir = path + dname + '256'
Query_dir = path + dname + '_Query.txt'
NB_CLS = 21

Query_set = Loader(Img_dir, Query_dir, NB_CLS)
Query_loader = torch.utils.data.DataLoader(Query_set, batch_size=args.JSCC_val_batch_size, num_workers=12)
valid_loader = Query_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

###########
#           JSCC model Class Config
###########

def get_generator(jscc_args, job_name):
    if jscc_args.JSCC_diversity:
        enc = EncoderCell(c_in=3, c_feat=jscc_args.JSCC_cfeat, c_out=jscc_args.JSCC_cout, attn=jscc_args.JSCC_adapt).to(jscc_args.device)
        dec = DecoderCell(c_in=jscc_args.JSCC_cout, c_feat=jscc_args.JSCC_cfeat, c_out=3, attn=jscc_args.JSCC_adapt).to(jscc_args.device)
        jscc_model = Div_model(jscc_args, enc, dec)
    else:
        enc = EncoderCell(c_in=3, c_feat=jscc_args.JSCC_cfeat, c_out=2*jscc_args.JSCC_cout, attn=jscc_args.JSCC_adapt).to(jscc_args.JSCC_device)
        dec = DecoderCell(c_in=2*jscc_args.JSCC_cout, c_feat=jscc_args.JSCC_cfeat, c_out=3, attn=jscc_args.JSCC_adapt).to(jscc_args.JSCC_device)
        if jscc_args.JSCC_res:
            res = EQUcell(6*jscc_args.JSCC_Nr, 128, 4).to(jscc_args.device)
            jscc_model = Mul_model(jscc_args, enc, dec, res)
        else:
            jscc_model = Mul_model(jscc_args, enc, dec)
    
    jscc_model = nn.DataParallel(jscc_model)  # Add this line to wrap model for DataParallel
    return jscc_model

def load_nets(job_name, nets):
    path = '{}/{}.pth'.format('models', job_name)
    
    if os.path.exists(path):
        checkpoint = torch.load(path)
        nets.load_state_dict(checkpoint['jscc_model'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        print(f"No checkpoint found at {path}")
        epoch = 0
    
    return epoch

def validate_model(generator_args, job_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    generator = get_generator(generator_args, job_name)
    
    # Load checkpoint
    epoch = load_nets(job_name, generator)
    print(f"Loaded checkpoint from epoch {epoch}")

    # Evaluation
    valid_loss, valid_aux = validate_epoch(generator_args, valid_loader, generator, epoch)

    print(f'Validation Loss: {valid_loss}')
    print(f'PSNR: {valid_aux["psnr"]}')
    print(f'SSIM: {valid_aux["ssim"]}')

def validate_epoch(args, loader, model, epoch):

    model.eval()

    loss_hist = []
    psnr_hist = []
    ssim_hist = []
    saved_images = False

    with torch.no_grad():
        with tqdm(loader, unit='batch') as tepoch:
            for _, (images, _) in enumerate(tepoch):

                epoch_postfix = OrderedDict()

                images = images.to(args.device).float()

                output = model(images, is_train=False)
                loss = nn.MSELoss()(output, images)

                epoch_postfix['l2_loss'] = '{:.4f}'.format(loss.item())

                ######  Predictions  ######
                predictions = torch.chunk(output, chunks=output.size(0), dim=0)
                target = torch.chunk(images, chunks=images.size(0), dim=0)

                ######  PSNR/SSIM/etc  ######
                psnr_vals = calc_psnr(predictions, target)
                psnr_hist.extend(psnr_vals)
                epoch_postfix['psnr'] = torch.mean(torch.tensor(psnr_vals)).item()

                ssim_vals = calc_ssim(predictions, target)
                ssim_hist.extend(ssim_vals)
                epoch_postfix['ssim'] = torch.mean(torch.tensor(ssim_vals)).item()
                
                # Show the snr/loss/psnr/ssim
                tepoch.set_postfix(**epoch_postfix)

                loss_hist.append(loss.item())

                # Save one pair of original and output images for visualization
                if not saved_images:
                    save_tensor_as_image(images[0], f'original_image_epoch_{epoch}.png')
                    save_tensor_as_image(output[0], f'output_image_epoch_{epoch}.png')
                    saved_images = True
            
            loss_mean = np.nanmean(loss_hist)

            psnr_hist = torch.tensor(psnr_hist)
            psnr_mean = torch.mean(psnr_hist).item()
            psnr_std = torch.sqrt(torch.var(psnr_hist)).item()

            ssim_hist = torch.tensor(ssim_hist)
            ssim_mean = torch.mean(ssim_hist).item()
            ssim_std = torch.sqrt(torch.var(ssim_hist)).item()

            predictions = torch.cat(predictions, dim=0)[:, [2, 1, 0]]
            target = torch.cat(target, dim=0)[:, [2, 1, 0]]

            return_aux = {'psnr': psnr_mean,
                          'ssim': ssim_mean,
                          'predictions': predictions,
                          'target': target,
                          'psnr_std': psnr_std,
                          'ssim_std': ssim_std}

    return loss_mean, return_aux

if __name__ == '__main__':
    job_name = 'train_JSCC_model_with_nuswide'
    validate_model(args, job_name)
