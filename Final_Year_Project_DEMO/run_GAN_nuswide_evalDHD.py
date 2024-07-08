import numpy as np 
import torch
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as LS

from JSCC_get_args import *
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


#########
#           Parameter Setting
#########
train_DHD = True

dname = 'nuswide'
path = './datasets/'
args = get_args_parser()

Img_dir = path + dname + '256'
Train_dir = path + dname + '_Train.txt'
Gallery_dir = path + dname + '_DB.txt'
Query_dir = path + dname + '_Query.txt'
org_size = 256
input_size = 224
NB_CLS = 21

AugS = Augmentation(org_size, 1.0)
AugT = Augmentation(org_size, 0.2)

Crop = nn.Sequential(Kg.CenterCrop(input_size))
Norm = nn.Sequential(Kg.Normalize(mean=torch.as_tensor([0.485, 0.456, 0.406]), std=torch.as_tensor([0.229, 0.224, 0.225])))

trainset = Loader(Img_dir, Train_dir, NB_CLS)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.DHD_batch_size, drop_last=True,
                                        shuffle=True, num_workers=args.DHD_num_workers)

Query_set = Loader(Img_dir, Query_dir, NB_CLS)
Query_loader = torch.utils.data.DataLoader(Query_set, batch_size=args.DHD_batch_size, num_workers=args.DHD_num_workers)
valid_loader = Query_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

######
#           DHD model Class Config (Discriminator)
######

def load_checkpoint(model, filename='checkpoint.pth.tar'):
    """Load checkpoint"""
    checkpoint = T.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint['epoch'], checkpoint['best_mAP']

class Hash_func(nn.Module):
    def __init__(self, fc_dim, N_bits, NB_CLS):
        super(Hash_func, self).__init__()
        self.Hash = nn.Sequential(
            nn.Linear(fc_dim, N_bits, bias=False),
            nn.LayerNorm(N_bits))
        self.P = nn.Parameter(torch.FloatTensor(NB_CLS, N_bits), requires_grad=True)
        nn.init.xavier_uniform_(self.P, gain=nn.init.calculate_gain('tanh'))

    def forward(self, X):
        X = self.Hash(X)
        return torch.tanh(X)

def get_discriminator(args):
    if args.DHD_encoder == 'AlexNet':
        Baseline = AlexNet()
        fc_dim = 4096
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

    H = Hash_func(fc_dim, args.DHD_N_bits, NB_CLS=21)
    net = nn.Sequential(Baseline, H)
    net = nn.DataParallel(net)  # Add this line to wrap model for DataParallel
    net.to(device)

    checkpoint_path = 'checkpoint.pth.tar'
    _, best_mAP = load_checkpoint(net, filename=checkpoint_path)
    print(f"Loaded checkpoint with best mAP: {best_mAP}")

    return net, H


###########
#           JSCC model Class Config (Generator)
###########
def save_checkpoint(state, is_best, output_dir, filename):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        T.save(state, os.path.join(output_dir, filename))  # save checkpoint
        print("=> Saving a new best model")

def get_generator(jscc_args, job_name,checkpoint_path):
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

    jscc_model = nn.DataParallel(jscc_model)  # Add this line to wrap model for DataParallel
    if checkpoint_path != None:
        # checkpoint_path = './models/train_JSCC_model_with_nuswide.pth'
        epoch = load_nets(checkpoint_path, jscc_model)
    return jscc_model


def train_gan(discriminator_args, generator_args, job_name,checkpoint_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = discriminator_args.DHD_gpu_id
    discriminator, H = get_discriminator(discriminator_args)
    generator = get_generator(generator_args, job_name,checkpoint_path)
    
    # Optimizers
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=discriminator_args.DHD_init_lr, weight_decay=10e-6)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=generator_args.JSCC_lr)
    
    # Schedulers
    d_scheduler = CosineAnnealingLR(d_optimizer, T_max=discriminator_args.DHD_max_epoch, eta_min=0)
    g_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(g_optimizer, lr_lambda=lambda x: 0.8)
    es = EarlyStopping(mode='min', min_delta=0, patience=discriminator_args.JSCC_train_patience)

    HP_criterion = HashProxy(discriminator_args.DHD_temp)
    HD_criterion = HashDistill()
    REG_criterion = BCEQuantization(discriminator_args.DHD_std)

    num_classes = 21

    MAX_mAP = 0.0
    mAP = 0.0

    AugT = Augmentation(256, discriminator_args.DHD_transformation_scale)  # Assuming org_size is 32 for CIFAR-10
    Crop = nn.Sequential(Kg.CenterCrop(224))
    Norm = nn.Sequential(Kg.Normalize(mean=torch.as_tensor([0.485, 0.456, 0.406]), std=torch.as_tensor([0.229, 0.224, 0.225])))

    n_critic = 1
    max_epoch = discriminator_args.DHD_max_epoch

    # evaluation metrics for JSON file
    evaluation_metrics = []

    # time stamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_filename = f'LOG/{job_name}_evaluation_metrics_{timestamp}.json'
    evaluation_json_filename = f'LOG/{job_name}_final_evaluation_{timestamp}.json'

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
            if train_DHD == False:
                discriminator.eval()
            else:
                discriminator.train()
            # with torch.no_grad():
            # Train Discriminator
            fake_inputs = generator(inputs, is_train=False).detach()  # detach to prevent gradients from flowing back to the generator

            l1 = torch.tensor(0., device=device)
            l2 = torch.tensor(0., device=device)
            l3 = torch.tensor(0., device=device)

            Is = Norm(Crop(fake_inputs))
            It = Norm(Crop(inputs))

            Xt = discriminator(It)
            l1 = HP_criterion(Xt, H.P, labels)

            Xs = discriminator(Is)
            l2 = HD_criterion(Xs, Xt) 
            l3 = REG_criterion(Xt) 

            if train_DHD:
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
                HD_loss = HD_criterion(discriminator(Norm(Crop(fake_inputs))), Xt.detach())
                g_loss = reconstruction_loss + HD_loss * discriminator_args.DHD_lambda1
                g_loss.backward()
                g_optimizer.step()

            if (i + 1) % 500 == 0:
                print(f'Step {i + 1}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

        valid_loss, valid_aux = validate_epoch(discriminator_args, valid_loader, generator)
        print("The Cosine Distance between Original and Transmitted Image is:", HD_loss.item())

        g_scheduler.step()

        if (epoch + 1) % discriminator_args.DHD_eval_epoch == 0 and (epoch + 1) >= discriminator_args.DHD_eval_init and train_DHD == True:
            mAP = DoRetrieval(device, discriminator.eval(), "./datasets/nuswide256", "./datasets/nuswide_DB.txt", "./datasets/nuswide_Query.txt", num_classes, 5000, discriminator_args)
            mAP_value = mAP.item()  # convert tensor to a Python number
            if mAP_value > MAX_mAP:
                MAX_mAP = mAP_value
                save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': discriminator.state_dict(),
                'best_mAP': MAX_mAP,
                'optimizer': d_optimizer.state_dict(),
                }, True, generator_args.DHD_output_dir, filename=job_name+".pth.tar")
                print("SAVE THE BEST CHECKPOINT of DHD")
            print("mAP: ", mAP_value, "MAX mAP: ", MAX_mAP)

        # Save evaluation results in every epoch
        evaluation_metrics.append({
            'epoch': epoch,
            'valid_loss': valid_loss.item(),  
            'PSNR': valid_aux['psnr'], 
            'SSIM': valid_aux['ssim'],
            'DHD_MAXmap': MAX_mAP,
            'mAP': mAP_value,
            "HD_loss":HD_loss.item()
        })

        # Writing result into Json file
        with open(json_filename, 'w') as f:
            json.dump(evaluation_metrics, f, indent=4)

        flag, best, best_epoch, bad_epochs = es.step(torch.tensor([valid_loss]), epoch)
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

    # Save final epoch checkpoint
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': discriminator.state_dict(),
        'best_mAP': MAX_mAP,
        'optimizer': d_optimizer.state_dict(),
        }, True, generator_args.DHD_output_dir, filename=job_name+"_last_epoch.pth.tar")
    print("SAVE FINAL EPOCH CHECKPOINT of DHD")

def validate_epoch(args, loader, model):

    model.eval()

    loss_hist = []
    psnr_hist = []
    ssim_hist = []

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
    job_name = 'TRAINboth_JSCC(tarined_withfixedDHD)_and_DHD_0627'
    args1 = args
    # evaluate_JSCC_path1 = './models/train_JSCC_model_with_nuswide.pth'
    evaluate_JSCC_path2 = './models/JSCC_model_DEMO_test_study_with_DHD.pth'
    distance1 = train_gan(args, args1, job_name,evaluate_JSCC_path2)
    # distance2 = evaluate_gan(args, args1, job_name,evaluate_JSCC_path2)
    # plot_hash_distances(distance1, distance2)
