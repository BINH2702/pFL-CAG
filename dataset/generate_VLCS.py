import time
import numpy as np
import os
import random
import torchvision.transforms as transforms
from utils.dataset_utils import split_data, save_file
from os import path
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_vlcs_data(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            data_path = path.join(dataset_path, data_path)
            label = int(label)
            data_paths.append(data_path)
            data_labels.append(label)
    return data_paths, data_labels

class VLCS(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(VLCS, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


def get_vlcs_dloader(dataset_path, domain_name):
    train_data_paths, train_data_labels = read_vlcs_data(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_vlcs_data(dataset_path, domain_name, split="test")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataset = VLCS(train_data_paths, train_data_labels, transforms_train, domain_name)
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dataset = VLCS(test_data_paths, test_data_labels, transforms_test, domain_name)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_loader, test_loader


random.seed(1)
np.random.seed(1)
data_path = "VLCS/"
dir_path = "VLCS/"

def generate_VLCS(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = data_path

    domains = ['caltech101', 'labelme', 'sun09', 'voc2007']

    X, y = [], []
    for d in domains:
        train_loader, test_loader = get_vlcs_dloader(root, d)

        for _, tt in enumerate(train_loader):
            train_data, train_label = tt
        for _, tt in enumerate(test_loader):
            test_data, test_label = tt


        dataset_image = []
        dataset_label = []

        dataset_image.extend(train_data.cpu().detach().numpy())
        dataset_image.extend(test_data.cpu().detach().numpy())
        dataset_label.extend(train_label.cpu().detach().numpy())
        dataset_label.extend(test_label.cpu().detach().numpy())

        X.append(np.array(dataset_image))
        y.append(np.array(dataset_label))

    labelss = []
    for yy in y:
        labelss.append(len(set(yy)))
    num_clients = len(y)
    print(f'Number of labels: {labelss}')
    print(f'Number of clients: {num_clients}')

    statistic = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))


    train_data, test_data = split_data(X, y)
    # modify the code in YOUR_ENV/lib/python3.8/site-packages/numpy/lib Line #678 from protocol=3 to protocol=4
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, max(labelss), 
        statistic, None, None, None)


if __name__ == "__main__":
    generate_VLCS(dir_path)