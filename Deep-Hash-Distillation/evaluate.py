import torch as T
import torch.nn as nn
from models import *
from loss import *
from Dataloader import Loader
from Retrieval import DoRetrieval
import kornia.augmentation as Kg
import matplotlib.pyplot as plt
import numpy as np

def save_tensor_as_image(tensor, filename):
    # Convert tensor to PIL image
    transform = transforms.ToPILImage()
    img = transform(tensor.cpu())
    
    # Save image
    img.save(filename)

def get_args_parser():
    parser = argparse.ArgumentParser('DHD', add_help=False)

    parser.add_argument('--gpu_id', default="0", type=str, help="""Define GPU id.""")
    parser.add_argument('--data_dir', default="data", type=str, help="""Path to dataset.""")
    parser.add_argument('--dataset', default="nuswide", type=str, help="""Dataset name: imagenet, nuswide_m, coco.""")
    
    parser.add_argument('--batch_size', default=128, type=int, help="""Training mini-batch size.""")
    parser.add_argument('--num_workers', default=12, type=int, help="""Number of data loading workers per GPU.""")
    parser.add_argument('--encoder', default="AlexNet", type=str, help="""Encoder network: ResNet, AlexNet, ViT, DeiT, SwinT.""")
    parser.add_argument('--N_bits', default=64, type=int, help="""Number of bits to retrieval.""")
    parser.add_argument('--init_lr', default=3e-4, type=float, help="""Initial learning rate.""")
    parser.add_argument('--warm_up', default=10, type=int, help="""Learning rate warm-up end.""")
    parser.add_argument('--lambda1', default=0.1, type=float, help="""Balancing hyper-paramter on self knowledge distillation.""")
    parser.add_argument('--lambda2', default=0.1, type=float, help="""Balancing hyper-paramter on bce quantization.""")
    parser.add_argument('--std', default=0.5, type=float, help="""Gaussian estimator standrad deviation.""")
    parser.add_argument('--temp', default=0.2, type=float, help="""Temperature scaling parameter on hash proxy loss.""")
    parser.add_argument('--transformation_scale', default=0.2, type=float, help="""Transformation scaling for self teacher: AlexNet=0.2, else=0.5.""")

    parser.add_argument('--max_epoch', default=500, type=int, help="""Number of epochs to train.""")
    parser.add_argument('--eval_epoch', default=10, type=int, help="""Compute mAP for Every N-th epoch.""")
    parser.add_argument('--eval_init', default=50, type=int, help="""Compute mAP after N-th epoch.""")
    parser.add_argument('--output_dir', default=".", type=str, help="""Path to save logs and checkpoints.""")

    return parser

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

def load_checkpoint(model, filename='checkpoint.pth.tar'):
    """Load checkpoint"""
    checkpoint = T.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint['epoch'], checkpoint['best_mAP']

def evaluate_model(args, checkpoint_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = T.device('cuda')

    path = "./" + args.data_dir + "/"
    dname = args.dataset

    if dname == 'imagenet':
        NB_CLS = 100
        Top_N = 1000
    elif dname == 'nuswide':
        NB_CLS = 21
        Top_N = 5000
    elif dname == 'nuswide256':
        NB_CLS = 21
        Top_N = 5000
    elif dname == 'coco':
        NB_CLS = 80
        Top_N = 5000
    else:
        print("Wrong dataset name.")
        return

    Img_dir = path + dname + '256'
    Gallery_dir = path + dname + '_DB.txt'
    Query_dir = path + dname + '_Query.txt'
    org_size = 256
    input_size = 224
    AugT = Augmentation(org_size, args.transformation_scale)

    Crop = nn.Sequential(Kg.CenterCrop(input_size))
    Norm = nn.Sequential(Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225])))

    Gallery_set = Loader(Img_dir, Gallery_dir, NB_CLS)
    Query_set = Loader(Img_dir, Query_dir, NB_CLS)

    Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers)
    Query_loader = T.utils.data.DataLoader(Query_set, batch_size=args.batch_size, num_workers=args.num_workers)

    HD_criterion = HashDistill()

    if args.encoder == 'AlexNet':
        Baseline = AlexNet()
        fc_dim = 4096
    elif args.encoder == 'ResNet':
        Baseline = ResNet()
        fc_dim = 2048
    elif args.encoder == 'ViT':
        Baseline = ViT('vit_base_patch16_224')
        fc_dim = 768
    elif args.encoder == 'DeiT':
        Baseline = DeiT('deit_base_distilled_patch16_224')
        fc_dim = 768
    elif args.encoder == 'SwinT':
        Baseline = SwinT('swin_base_patch4_window7_224')
        fc_dim = 1024
    else:
        print("Wrong encoder name.")
        return

    H = Hash_func(fc_dim, args.N_bits, NB_CLS)
    net = nn.Sequential(Baseline, H)
    net = nn.DataParallel(net)
    net.to(device)

    # Load the checkpoint
    _, best_mAP = load_checkpoint(net, filename=checkpoint_path)
    print(f"Loaded checkpoint with best mAP: {best_mAP}")

    net.eval()

    with T.no_grad():
        gallery_codes, gallery_labels = [], []
        for i, data in enumerate(Gallery_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = Norm(Crop(inputs))
            outputs = net(inputs)
            gallery_codes.append(outputs)
            gallery_labels.append(labels)

        gallery_codes = T.cat(gallery_codes, dim=0)
        gallery_labels = T.cat(gallery_labels, dim=0)

        query_codes, query_labels = [], []
        hash_distance_list = []
        hamming_distance_list = []
        for i, data in enumerate(Query_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = Norm(Crop(inputs))
            transformed_inputs = Norm(Crop(AugT(inputs)))
            outputs = net(inputs)
            transformed_outputs = net(transformed_inputs)
            hash_distance_list.append(HD_criterion(outputs, transformed_outputs).cpu().numpy())

            # Calculate the binary hash codes
            binary_outputs = T.sign(outputs)
            binary_transformed_outputs = T.sign(transformed_outputs)

            # # Calculate Hamming distance
            # hamming_dist = (binary_outputs.shape[1] - (binary_outputs @ binary_transformed_outputs.t())).float().sum(dim=1).cpu().numpy()
            # hamming_distance_list.append(hamming_dist)

            query_codes.append(outputs)
            query_labels.append(labels)

        query_codes = T.cat(query_codes, dim=0)
        query_labels = T.cat(query_labels, dim=0)

    gallery_codes = T.sign(gallery_codes)
    query_codes = T.sign(query_codes)

    # Convert lists to numpy arrays
    hash_distances = np.array(hash_distance_list)
    # import pdb; pdb.set_trace()
    # hamming_distances = np.array(hamming_distance_list)

    # Calculate average distances
    average_hash_distance = np.mean(hash_distances)
    # average_hamming_distance = np.mean(hamming_distances)

    # Perform retrieval
    import pdb; pdb.set_trace()
    mAP = DoRetrieval(device, net, Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args)
    print(f"mAP of the model: {mAP}")
    print(f"Average hash distance: {average_hash_distance}")
    # print(f"Average Hamming distance: {average_hamming_distance}")

    # Plot the hash distances
    plt.figure(figsize=(12, 6))

    # plt.subplot(1, 2, 1)
    plt.hist(hash_distances, bins=50, alpha=0.75, color='blue')
    plt.title('Distribution of Hash Distances')
    plt.xlabel('Hash Distance')
    plt.ylabel('Frequency')

    # plt.subplot(1, 2, 2)
    # plt.hist(hamming_distances, bins=50, alpha=0.75, color='red')
    # plt.title('Distribution of Hamming Distances')
    # plt.xlabel('Hamming Distance')
    # plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig("distances.png")

    import pdb; pdb.set_trace()
    save_tensor_as_image(inputs, 'last_inputs.png')
    save_tensor_as_image(transformed_inputs, 'last_transformed_inputs.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DHD', parents=[get_args_parser()])
    args = parser.parse_args()
    checkpoint_path = 'checkpoint.pth.tar'  # Replace with your checkpoint path
    evaluate_model(args, checkpoint_path)
