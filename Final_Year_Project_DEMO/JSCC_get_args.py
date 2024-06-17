import argparse

# def get_args_parser():
#     parser = argparse.ArgumentParser('DHD', add_help=False)

#     parser.add_argument('--gpu_id', default="0", type=str, help="""Define GPU id.""")
#     parser.add_argument('--data_dir', default="/data", type=str, help="""Path to dataset.""")
#     parser.add_argument('--dataset', default="imagenet", type=str, help="""Dataset name: imagenet, nuswide_m, coco.""")
    
#     parser.add_argument('--batch_size', default=128, type=int, help="""Training mini-batch size.""")
#     parser.add_argument('--num_workers', default=12, type=int, help="""Number of data loading workers per GPU.""")
#     parser.add_argument('--encoder', default="AlexNet", type=str, help="""Encoder network: ResNet, AlexNet, ViT, DeiT, SwinT.""")
#     parser.add_argument('--N_bits', default=64, type=int, help="""Number of bits to retrieval.""")
#     parser.add_argument('--init_lr', default=3e-4, type=float, help="""Initial learning rate.""")
#     parser.add_argument('--warm_up', default=10, type=int, help="""Learning rate warm-up end.""")
#     parser.add_argument('--lambda1', default=0.1, type=float, help="""Balancing hyper-paramter on self knowledge distillation.""")
#     parser.add_argument('--lambda2', default=0.1, type=float, help="""Balancing hyper-paramter on bce quantization.""")
#     parser.add_argument('--std', default=0.5, type=float, help="""Gaussian estimator standrad deviation.""")
#     parser.add_argument('--temp', default=0.2, type=float, help="""Temperature scaling parameter on hash proxy loss.""")
#     parser.add_argument('--transformation_scale', default=0.2, type=float, help="""Transformation scaling for self teacher: AlexNet=0.2, else=0.5.""")

#     parser.add_argument('--max_epoch', default=500, type=int, help="""Number of epochs to train.""")
#     parser.add_argument('--eval_epoch', default=10, type=int, help="""Compute mAP for Every N-th epoch.""")
#     parser.add_argument('--eval_init', default=50, type=int, help="""Compute mAP after N-th epoch.""")
#     parser.add_argument('--output_dir', default=".", type=str, help="""Path to save logs and checkpoints.""")

#     return parser


def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', default  = 'cifar')

    # Neural Network setting
    parser.add_argument('-cout', type=int, default  = 12)
    parser.add_argument('-cfeat', type=int, default  = 256)

    # The transmitter setting
    parser.add_argument('-distribute', default  = False)
    parser.add_argument('-res', default  = True)
    parser.add_argument('-diversity', default  = True)
    parser.add_argument('-adapt', default  = True)
    parser.add_argument('-Nt',  default  = 2)
    parser.add_argument('-P1',  default  = 10.0)
    parser.add_argument('-P2',  default  = 10.0)
    parser.add_argument('-P1_rng',  default  = 4.0)
    parser.add_argument('-P2_rng',  default  = 4.0)

    # The receiver setting
    parser.add_argument('-Nr',  default  = 2)

    # training setting
    parser.add_argument('-epoch', type=int, default  = 400)
    parser.add_argument('-lr', type=float, default  = 1e-4)
    parser.add_argument('-train_patience', type=int, default  = 12)
    parser.add_argument('-train_batch_size', type=int, default  = 32)

    parser.add_argument('-val_batch_size', type=int, default  = 32)
    parser.add_argument('-resume', default  = False)
    parser.add_argument('-path', default  = 'models/')


    return args
