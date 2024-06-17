import numpy as np 
import torch
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as LS

from JSCC_get_args import get_args
from JSCC_modules import *
from dataset import CIFAR10, ImageNet, Kodak
from JSCC_utils import *
from JSCC_coop_network import *

from DHD_models import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from DHD_loss import *
from DHD_Retrieval import DoRetrieval
from DHD_modifiedALEX import ModifiedAlexNet
from Dataloader import Loader

from datetime import datetime
import json


#########
#   Parser for Discriminator
#########

def get_args_parser():
    parser = argparse.ArgumentParser()

    ### Arguments for DHD (Discriminator)
    parser.add_argument('--DHD_gpu_id', default="0", type=str, help="""Define GPU id.""")
    parser.add_argument('--DHD_data_dir', default="/datasets", type=str, help="""Path to dataset.""")
    parser.add_argument('--DHD_dataset', default="imagenet", type=str, help="""Dataset name: imagenet, nuswide_m, coco.""")
    
    parser.add_argument('--DHD_batch_size', default=128, type=int, help="""Training mini-batch size.""")
    parser.add_argument('--DHD_num_workers', default=12, type=int, help="""Number of data loading workers per GPU.""")
    parser.add_argument('--DHD_encoder', default="AlexNet", type=str, help="""Encoder network: ResNet, AlexNet, ViT, DeiT, SwinT.""")
    parser.add_argument('--DHD_N_bits', default=64, type=int, help="""Number of bits to retrieval.""")
    parser.add_argument('--DHD_init_lr', default=3e-4, type=float, help="""Initial learning rate.""")
    parser.add_argument('--DHD_warm_up', default=10, type=int, help="""Learning rate warm-up end.""")
    parser.add_argument('--DHD_lambda1', default=0.1, type=float, help="""Balancing hyper-paramter on self knowledge distillation.""")
    parser.add_argument('--DHD_lambda2', default=0.1, type=float, help="""Balancing hyper-paramter on bce quantization.""")
    parser.add_argument('--DHD_std', default=0.5, type=float, help="""Gaussian estimator standrad deviation.""")
    parser.add_argument('--DHD_temp', default=0.2, type=float, help="""Temperature scaling parameter on hash proxy loss.""")
    parser.add_argument('--DHD_transformation_scale', default=0.2, type=float, help="""Transformation scaling for self teacher: AlexNet=0.2, else=0.5.""")

    parser.add_argument('--DHD_max_epoch', default=500, type=int, help="""Number of epochs to train.""")
    parser.add_argument('--DHD_eval_epoch', default=1, type=int, help="""Compute mAP for Every N-th epoch.""")
    parser.add_argument('--DHD_eval_init', default=1, type=int, help="""Compute mAP after N-th epoch.""")
    parser.add_argument('--DHD_output_dir', default=".", type=str, help="""Path to save logs and checkpoints.""")

    ### Arguments for JSCC (Generator)
    parser.add_argument('-JSCC_dataset', default  = 'cifar')

    # Neural Network setting
    parser.add_argument('-JSCC_cout', type=int, default  = 12)
    parser.add_argument('-JSCC_cfeat', type=int, default  = 256)

    # The transmitter setting
    parser.add_argument('-JSCC_distribute', default  = False)
    parser.add_argument('-JSCC_res', default  = True)
    parser.add_argument('-JSCC_diversity', default  = True)
    parser.add_argument('-JSCC_adapt', default  = True)
    parser.add_argument('-JSCC_Nt',  default  = 2)
    parser.add_argument('-JSCC_P1',  default  = 10.0)
    parser.add_argument('-JSCC_P2',  default  = 10.0)
    parser.add_argument('-JSCC_P1_rng',  default  = 4.0)
    parser.add_argument('-JSCC_P2_rng',  default  = 4.0)

    # The receiver setting
    parser.add_argument('-JSCC_Nr',  default  = 2)

    # training setting
    parser.add_argument('-JSCC_epoch', type=int, default  = 400)
    parser.add_argument('-JSCC_lr', type=float, default  = 1e-4)
    parser.add_argument('-JSCC_train_patience', type=int, default  = 12)
    parser.add_argument('-JSCC_train_batch_size', type=int, default  = 32)

    parser.add_argument('-JSCC_val_batch_size', type=int, default  = 32)
    parser.add_argument('-JSCC_resume', default  = False)
    parser.add_argument('-JSCC_path', default  = 'models/')

    args = parser.parse_args()

    return args


#########
#           Parameter Setting
#########

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = get_args_parser()
args.device = device

job_name = 'JSCC_dist_'+str(args.JSCC_distribute)+'_div_'+str(args.JSCC_diversity)+'_Nr_'+str(args.JSCC_Nr)+'_dataset_'+str(args.JSCC_dataset)+'_cout_'+str(args.JSCC_cout)\
                +'_P1_' + str(args.JSCC_P1) + '_P2_' + str(args.JSCC_P2) +'_is_adapt_'+ str(args.JSCC_adapt) + '_is_res_' +str(args.JSCC_res)
if args.JSCC_adapt:
    job_name = job_name + '_P1_rng_' + str(args.JSCC_P1_rng) + '_P2_rng_' + str(args.JSCC_P2_rng)

print(args)
print(job_name)

frame_size = (32, 32)
src_ratio = args.JSCC_cout / (3*4*4)

train_set = CIFAR10('datasets/cifar-10-batches-py', 'TRAIN')
valid_set = CIFAR10('datasets/cifar-10-batches-py', 'VALIDATE')
eval_set = CIFAR10('datasets/cifar-10-batches-py', 'EVALUATE')

######
#           DHD model Class Config (Discriminator)
######

class Hash_func(nn.Module):
    def __init__(self, fc_dim, N_bits, NB_CLS):
        super(Hash_func, self).__init__()
        self.Hash = nn.Sequential(
            nn.Linear(fc_dim, N_bits, bias=False),
            nn.LayerNorm(N_bits))
        self.P = nn.Parameter(T.FloatTensor(NB_CLS, N_bits), requires_grad=True)
        nn.init.xavier_uniform_(self.P, gain=nn.init.calculate_gain('tanh'))

    def forward(self, X):
        X = self.Hash(X)
        return T.tanh(X)

def get_discriminator(args):
    if args.DHD_encoder == 'AlexNet':
        Baseline = ModifiedAlexNet()
        fc_dim = 10
    elif args.DHD_encoder == 'ResNet':
        Baseline = ResNet()
        fc_dim = 2048
    elif args.DHD_encoder == 'ViT':
        Baseline = ViT('vit_base_patch16_224')
        fc_dim = 768
    elif args.DHD_encoder == 'DeiT':
        Baseline = DeiT('deit_base_distilled_patch16_224')
        fc_dim = 768
    elif args.DHD_encoder == 'SwinT':
        Baseline = SwinT('swin_base_patch4_window7_224')
        fc_dim = 1024
    else:
        print("Wrong encoder name.")
        return None

    H = Hash_func(fc_dim, args.DHD_N_bits, NB_CLS = 10)
    net = nn.Sequential(Baseline, H)
    net.to(device)
    return net,H


###########
#           JSCC model Class Config (Generator)
###########

def get_generator(jscc_args,job_name):
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
    
    if jscc_args.JSCC_resume:
        load_weights(job_name, jscc_model)

    return jscc_model


# solver = optim.Adam(jscc_model.parameters(), lr=args.lr)
# scheduler = LS.MultiplicativeLR(solver, lr_lambda=lambda x: 0.8)
# es = EarlyStopping(mode='min', min_delta=0, patience=args.train_patience)

###### Dataloader
train_loader = data.DataLoader(
    dataset=train_set,
    batch_size=args.JSCC_train_batch_size,
    shuffle=True,
    num_workers=2
        )

valid_loader = data.DataLoader(
    dataset=valid_set,
    batch_size=args.JSCC_val_batch_size,
    shuffle=True,
    num_workers=2
        )

eval_loader = data.DataLoader(
    dataset=eval_set,
    batch_size=args.JSCC_val_batch_size,
    shuffle=True,
    num_workers=2
)


def train_epoch(loader, model, solvers):

    model.train()

    with tqdm(loader, unit='batch') as tepoch:
        for _, (images, _) in enumerate(tepoch):
            
            epoch_postfix = OrderedDict()

            images = images.to(args.device).float()
            
            solvers.zero_grad()
            output = model(images, is_train = True)

            loss = nn.MSELoss()(output, images)
            loss.backward()
            solvers.step()

            epoch_postfix['l2_loss'] = '{:.4f}'.format(loss.item())

            tepoch.set_postfix(**epoch_postfix)

# 训练函数
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import kornia.augmentation as Kg

# 训练函数
def train_gan(discriminator_args, generator_args, job_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = discriminator_args.DHD_gpu_id
    discriminator, H = get_discriminator(discriminator_args)
    generator = get_generator(generator_args, job_name)
    
    # Optimizers
    d_optimizer = T.optim.Adam(discriminator.parameters(), lr=discriminator_args.DHD_init_lr, weight_decay=10e-6)
    g_optimizer = T.optim.Adam(generator.parameters(), lr=generator_args.JSCC_lr)
    
    # Schedulers
    d_scheduler = CosineAnnealingLR(d_optimizer, T_max=discriminator_args.DHD_max_epoch, eta_min=0)
    g_scheduler = T.optim.lr_scheduler.MultiplicativeLR(g_optimizer, lr_lambda=lambda x: 0.8)
    es = EarlyStopping(mode='min', min_delta=0, patience=discriminator_args.JSCC_train_patience)

    HP_criterion = HashProxy(discriminator_args.DHD_temp)
    HD_criterion = HashDistill()
    REG_criterion = BCEQuantization(discriminator_args.DHD_std)

    num_classes = 10

    MAX_mAP = 0.0
    mAP = 0.0

    AugT = Augmentation(32, discriminator_args.DHD_transformation_scale)  # Assuming org_size is 32 for CIFAR-10
    Crop = nn.Sequential(Kg.CenterCrop(32))
    Norm = nn.Sequential(Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225])))

    n_critic = 1
    max_epoch = discriminator_args.DHD_max_epoch

    # evaluation metrics for JSON file
    evaluation_metrics = []

    # time stamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_filename = f'.\LOG\{job_name}_evaluation_metrics_{timestamp}.json'
    evaluation_json_filename = f'.\LOG\{job_name}_final_evaluation_{timestamp}.json'

    # Initial epoch
    epoch = 0

    while epoch < generator_args.JSCC_epoch and not generator_args.JSCC_resume:
        epoch += 1
        print(f'Epoch {epoch}/{max_epoch}')
        C_loss = 0.0
        S_loss = 0.0
        R_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Train Discriminator
            fake_inputs = generator(inputs, is_train=False).detach()  # detach to prevent gradients from flowing back to the generator

            l1 = T.tensor(0., device=device)
            l2 = T.tensor(0., device=device)
            l3 = T.tensor(0., device=device)

            Is = Norm(Crop(fake_inputs))
            It = Norm(Crop(AugT(inputs)))

            Xt = discriminator(It)
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            l1 = HP_criterion(Xt, H.P, one_hot_labels)

            Xs = discriminator(Is)
            l2 = HD_criterion(Xs, Xt) * discriminator_args.DHD_lambda1
            l3 = REG_criterion(Xt) * discriminator_args.DHD_lambda2

            d_loss = l1 + l2 + l3
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)  # Retain graph to allow second backward pass
            d_optimizer.step()

            C_loss += l1.item()
            S_loss += l2.item()
            R_loss += l3.item()

            # Train Generator
            if i % n_critic == 0:
                g_optimizer.zero_grad()
                fake_inputs = generator(inputs, is_train=True)  # regenerate fake inputs with gradients
                reconstruction_loss = nn.MSELoss()(fake_inputs, inputs)
                g_loss = reconstruction_loss + HD_criterion(discriminator(Norm(Crop(fake_inputs))), Xt.detach()) * discriminator_args.DHD_lambda1
                g_loss.backward()
                g_optimizer.step()

            if (i + 1) % 500 == 0:
                print(f'Step {i + 1}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

        valid_loss, valid_aux = validate_epoch(discriminator_args, valid_loader, generator)

        d_scheduler.step()
        g_scheduler.step()

        if (epoch + 1) % discriminator_args.DHD_eval_epoch == 0 and (epoch + 1) >= discriminator_args.DHD_eval_init:
            mAP = DoRetrieval(device, discriminator.eval(), discriminator_args.DHD_data_dir, discriminator_args.DHD_dataset+'_DB.txt', discriminator_args.DHD_dataset+'_Query.txt', num_classes, discriminator_args.DHD_Top_N, discriminator_args)
            if mAP > MAX_mAP:
                MAX_mAP = mAP
            print("mAP: ", mAP, "MAX mAP: ", MAX_mAP)
            discriminator.train()

        # Save evaluation results in every epoch
        evaluation_metrics.append({
            'epoch': epoch,
            'valid_loss': valid_loss.item(),  
            'PSNR':valid_aux['psnr'], 
            'SSIM':valid_aux['ssim'],
            'DHD_MAXmap':MAX_mAP,
            'mAP':mAP
        })

        # Writing result into Json file
        with open(json_filename, 'w') as f:
            json.dump(evaluation_metrics, f, indent=4)

        flag, best, best_epoch, bad_epochs = es.step(T.tensor([valid_loss]), epoch)
        if flag:
            print('ES criterion met; loading best weights from epoch {}'.format(best_epoch))
            _ = load_weights(job_name, generator)
            break
        else:
            if bad_epochs == 0:
                print('average l2_loss: ', valid_loss.item())
                save_nets(job_name, generator, epoch)
                best_epoch = epoch
                print('saving best net weights...')
            elif bad_epochs % (es.patience // 3) == 0:
                g_scheduler.step()
                print('lr updated: {:.5f}'.format(g_scheduler.get_last_lr()[0]))

    print('evaluating...')
    print(job_name)

    final_evaluation_metrics = []

    for P in range(6, 16, 2):
        generator.P1, generator.P2 = P, P
        _, eval_aux = validate_epoch(discriminator_args, eval_loader, generator)
        print(eval_aux['psnr'])
        print(eval_aux['ssim'])
        
        final_evaluation_metrics.append({
            'psnr': eval_aux['psnr'],
            'ssim': eval_aux['ssim'],
        })

    with open(evaluation_json_filename, 'w') as f:
        json.dump(final_evaluation_metrics, f, indent=4)


def validate_epoch(args, loader, model):

    model.eval()

    loss_hist = []
    psnr_hist = []
    ssim_hist = []
    #msssim_hist = []

    with torch.no_grad():
        with tqdm(loader, unit='batch') as tepoch:
            for _, (images, _) in enumerate(tepoch):

                epoch_postfix = OrderedDict()

                images = images.to(args.device).float()

                output = model(images, is_train = False)
                #output = model(images)
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
    job_name = 'JSCC_model'  # 定义你的 job name
    args1 = args
    train_gan(args, args1, job_name)

