import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import copy


train_data = datasets.CIFAR10(
    root="complete-data",
    train=True,
    download=True,
    transform=ToTensor()
)
#train_data[i] is a tuple(33232, label) (C,H,W)
test_data = datasets.CIFAR10(
    root="complete-data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels={
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

def visualize_dataset(imgs):
    fig, axes = plt.subplots(3, 3)
    axes=axes.flatten()
    for i, img in enumerate(imgs):
        axes[i].imshow(img[0].permute(1,2,0)) #H,W,C
        axes[i].axis('off')
        axes[i].set_title(labels[img[1]])
    plt.tight_layout()
    plt.show()
    
def random_Crop(randInd):
    randCrop_data = datasets.CIFAR10(
    root="complete-data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(20)])
    )
    randCrop=[randCrop_data[i] for i in randInd]
    visualize_dataset(randCrop)

def rotated_90(randInd):
    rotate_data90 = datasets.CIFAR10(
        root="complete-data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.RandomRotation([89,90])])
    )
    rot90=[rotate_data90[i] for i in randInd]
    visualize_dataset(rot90) 

def rotated_180(randInd):
    rotate_data180 = datasets.CIFAR10(
        root="complete-data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.RandomRotation([179,180])])
    )
    rot180=[rotate_data180[i] for i in randInd]
    visualize_dataset(rot180)
    
def rotated_270(randInd):
    rotate_data270 = datasets.CIFAR10(
        root="complete-data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.RandomRotation([269,270])])
    )
    rot270=[rotate_data270[i] for i in randInd]
    visualize_dataset(rot270)

def grayscaled(randInd):
    gray = datasets.CIFAR10(
        root="complete-data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Grayscale(3)])
    )
    grays=[gray[i] for i in randInd]
    visualize_dataset(grays)

def weak_Perspective(randInd):
    perspective = datasets.CIFAR10(
        root="complete-data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.RandomPerspective(0.3, 1)])
    )
    per=[perspective[i] for i in randInd]
    visualize_dataset(per) 

def strong_Perspective(randInd):
    perspective2 = datasets.CIFAR10(
        root="complete-data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.RandomPerspective(0.6, 1)])
    )
    per2=[perspective2[i] for i in randInd]
    visualize_dataset(per2)

def combined_Augmented_Data(randInd):
    rand_imgs=[train_data[i] for i in rand_indices]
    visualize_dataset(rand_imgs) # This is normal images with no transformations
    
    random_Crop(randInd)
    rotated_90(randInd)
    rotated_180(randInd)
    rotated_270(randInd)
    grayscaled(randInd)
    weak_Perspective(randInd)
    strong_Perspective(randInd)
    

rand_indices=np.random.randint(0,len(train_data),size=9)
a = combined_Augmented_Data(rand_indices)

