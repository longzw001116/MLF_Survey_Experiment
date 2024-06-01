import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MyDataset(Dataset):

    def __init__(self, path, hw) :
        super().__init__()

        self.imgs_list = []
        self.data_transform = transforms.Compose([
            transforms.Resize((hw, hw)),
            transforms.ToTensor()
        ])

        for file in os.listdir(path):
            self.imgs_list.append(os.path.join(path, file))

    def __getitem__(self, index):
        image_path = self.imgs_list[index]
        img = Image.open(image_path)
        img = self.data_transform(img)
        return img
    
    def __len__(self):
        return len(self.imgs_list)