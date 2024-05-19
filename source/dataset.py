import random
import torch
import torch.utils
from torchvision import transforms, datasets
from setup import setup_config

class StarbucksDataset():
    def __init__(self):
        self.seed = setup_config()['SEED']
        self.batch_size = setup_config()['BATCH_SIZE']
        self.size = setup_config()["IMAGE_SIZE"]
    
    def trans(self):
        data_transforms=transforms.Compose([
            transforms.Resize(self.size),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        return data_transforms
    

    def splitdata(self):
        random.seed(self.seed)
        image_datasets = datasets.ImageFolder(root='./data/starbucks_top10', transform=self.trans())

        train_size = int(0.4*len(image_datasets))
        val_size = int(0.3*len(image_datasets))
        test_size = len(image_datasets) - (train_size + val_size)

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(image_datasets, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(self.seed))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader