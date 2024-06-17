import torch as T
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import kornia.augmentation as Kg
from Dataloader import *
import numpy as np

def save_tensor_as_image(tensor, filename):
    # Convert tensor to PIL image
    transform = transforms.ToPILImage()
    img = transform(tensor.cpu())
    
    # Save image
    img.save(filename)

def visualize_images(images, titles, filename):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 20))
    for img, title, ax in zip(images, titles, axes):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.savefig(filename)
    plt.close()

def Evaluate_mAP(device, gallery_codes, query_codes, gallery_labels, query_labels, Top_N=None):
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        retrieval = (query_labels[i, :] @ gallery_labels.t() > 0).float()
        hamming_dist = (gallery_codes.shape[1] - query_codes[i, :] @ gallery_codes.t())

        retrieval = retrieval[T.argsort(hamming_dist)][:Top_N]
        retrieval_cnt = retrieval.sum().int().item()

        if retrieval_cnt == 0:
            continue

        score = T.linspace(1, retrieval_cnt, retrieval_cnt).to(device)
        index = (T.nonzero(retrieval == 1, as_tuple=False).squeeze() + 1.0).float().to(device)

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP

def DoRetrieval(device, net, Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args):
    Gallery_set = Loader(Img_dir, Gallery_dir, NB_CLS)
    Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers)

    Query_set = Loader(Img_dir, Query_dir, NB_CLS)
    Query_loader = T.utils.data.DataLoader(Query_set, batch_size=args.batch_size, num_workers=args.num_workers)

    Crop_Normalize = T.nn.Sequential(
        Kg.CenterCrop(224),
        Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225]))
    )

    gallery_images = []
    query_images = []
    top_n_images = []

    with T.no_grad():
        for i, data in enumerate(Gallery_loader, 0):
            gallery_x_batch, gallery_y_batch = data[0].to(device), data[1].to(device)
            gallery_x_batch = Crop_Normalize(gallery_x_batch)
            if gallery_y_batch.dim() == 1:
                gallery_y_batch = T.eye(NB_CLS, device=device)[gallery_y_batch]

            outputs = net(gallery_x_batch)

            if i == 0:
                gallery_c = outputs
                gallery_y = gallery_y_batch
                gallery_images = gallery_x_batch
            else:
                gallery_c = T.cat([gallery_c, outputs], 0)
                gallery_y = T.cat([gallery_y, gallery_y_batch], 0)
                gallery_images = T.cat([gallery_images, gallery_x_batch], 0)

        for i, data in enumerate(Query_loader, 0):
            query_x_batch, query_y_batch = data[0].to(device), data[1].to(device)
            query_x_batch = Crop_Normalize(query_x_batch)
            if query_y_batch.dim() == 1:
                query_y_batch = T.eye(NB_CLS, device=device)[query_y_batch]

            outputs = net(query_x_batch)

            if i == 0:
                query_c = outputs
                query_y = query_y_batch
                query_images = query_x_batch
            else:
                query_c = T.cat([query_c, outputs], 0)
                query_y = T.cat([query_y, query_y_batch], 0)
                query_images = T.cat([query_images, query_x_batch], 0)

    gallery_c = T.sign(gallery_c)
    query_c = T.sign(query_c)

    mAP = Evaluate_mAP(device, gallery_c, query_c, gallery_y, query_y, Top_N)

    # Visualize top N images
    num_query = query_y.shape[0]
    for i in range(num_query):
        retrieval = (query_y[i, :] @ gallery_y.t() > 0).float()
        hamming_dist = (gallery_c.shape[1] - query_c[i, :] @ gallery_c.t())

        retrieval = retrieval[T.argsort(hamming_dist)][:Top_N]
        top_n_idx = T.argsort(hamming_dist)[:Top_N]

        retrieved_images = gallery_images[top_n_idx].cpu()
        query_image = query_images[i].cpu()

        images_to_show = [transforms.ToPILImage()(query_image)] + [transforms.ToPILImage()(retrieved_images[j]) for j in range(Top_N)]
        titles = ["Query"] + [f"Top {j+1}" for j in range(Top_N)]
        visualize_images(images_to_show, titles, f'results/query_{i}_top_{Top_N}.png')

    return mAP