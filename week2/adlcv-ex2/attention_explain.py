import numpy as np
import os
import cv2
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
from matplotlib import pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from vit import ViT

from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def select_two_classes_from_cifar10(dataset, classes):
    idx = (np.array(dataset.targets) == classes[0]) | (np.array(dataset.targets) == classes[1])
    dataset.targets = np.array(dataset.targets)[idx]
    dataset.targets[dataset.targets==classes[0]] = 0
    dataset.targets[dataset.targets==classes[1]] = 1
    dataset.targets= dataset.targets.tolist()
    dataset.data = dataset.data[idx]
    return dataset

def prepare_dataloaders(batch_size, classes=[3, 7]):
    # TASK: Experiment with data augmentation
    train_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    # select two classes
    trainset = select_two_classes_from_cifar10(trainset, classes=classes)
    testset = select_two_classes_from_cifar10(testset, classes=classes)

    # reduce dataset size
    trainset, _ = torch.utils.data.random_split(trainset, [5000, 5000])
    testset, _ = torch.utils.data.random_split(testset, [1000, 1000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False
    )
    return trainloader, testloader, trainset, testset

# def show_mask_on_image(img, mask):
#     img = np.float32(img) / 255
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     heatmap = heatmap[..., ::-1]
#     heatmap = np.float32(heatmap) / 255
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#     return np.uint8(255 * cam)

def show_mask_on_image(img, mask):
    #img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # bgr to rgb
    heatmap = heatmap[..., ::-1]
    heatmap = np.float32(heatmap) / 255
    cam =  0.8*heatmap +  np.float32(img)
    #cam = cam / np.max(cam)
    return cam

def main(image_size=(32,32), patch_size=(4,4), channels=3,
         embed_dim=128, num_heads=4, num_layers=8, num_classes=2,
         pos_enc='fixed', pool='cls', dropout=0.3, fc_dim=128,
         num_epochs=20, batch_size=16, lr=1e-4, warmup_steps=625,
         weight_decay=1e-3, gradient_clipping=1

    ):

    _, _, dataset, _ = prepare_dataloaders(batch_size=batch_size)

    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels,
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim,
                num_classes=num_classes
    )
    model.load_state_dict(torch.load("model.pth"))

    if torch.cuda.is_available():
        model = model.to('cuda')

    while True:
        # Get random image from dataset
        idx = random.randint(0, len(dataset))
        print("Index:", idx)
        img_tensor, label = dataset[idx]
        print("Label:", label)
        img_tensor = img_tensor.unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor = img_tensor.to('cuda')

        attention_rollout = VITAttentionRollout(model,
                                                attention_layer_name="attention",
                                                head_fusion="max",
                                                discard_ratio=0.9)
        mask, out = attention_rollout(img_tensor)

        print(out)
        s = F.softmax(out, dim=-1)
        print(s)
        l = s.argmax(dim=-1).item()
        # Check if the model is correct
        print("Model prediction: ", "correct" if l==label else "incorrect")

        np_img = img_tensor.cpu().numpy().squeeze(0)
        np_img = np_img.transpose(1, 2, 0)
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))

        plt.subplot(1, 3, 1)
        plt.imshow(np_img)
        plt.subplot(1, 3, 2)
        plt.imshow(mask)
        plt.subplot(1, 3, 3)
        mask = show_mask_on_image(np_img, mask)
        plt.imshow(mask)
        plt.show()


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")
    set_seed(seed=1)
    main()
