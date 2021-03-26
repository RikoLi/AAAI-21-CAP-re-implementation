from PIL import Image
import copy
import torchvision
import torch

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, img_shape, dataset):
        super(TrainDataset, self).__init__()
        self.img_shape = img_shape
        self.samples = copy.deepcopy(dataset.samples) # global dataset samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fname, vid, camid, pseudo_cluster_label, proxy_label, abs_proxy_label = self.samples[index]

        # convert str to int
        vid = int(vid.split('_')[-1])
        camid = int(camid.split('_')[-1])

        h, w = self.img_shape
        img = Image.open(fname).convert('RGB')

        # preprocessing
        img = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(h,w), interpolation=3),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.Pad(10),
            torchvision.transforms.RandomCrop(size=(h,w)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # SpCL config
            torchvision.transforms.RandomErasing()
        ])(img)

        return img, vid, camid, pseudo_cluster_label, proxy_label, abs_proxy_label

class GlobalDataset(torch.utils.data.Dataset):
    def __init__(self, img_shape, dataset):
        super(GlobalDataset, self).__init__()
        self.img_shape = img_shape
        self.samples = copy.deepcopy(dataset.train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fname, vid, camid = self.samples[index]

        # convert str to int
        vid = int(vid.split('_')[-1])
        camid = int(camid.split('_')[-1])

        h, w = self.img_shape
        img = Image.open(fname).convert('RGB')
        
        # preprocessing
        img = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(h,w), interpolation=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img)
        
        return img, fname, vid, camid