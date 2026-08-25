import torchvision.transforms as transforms
import torchvision.transforms.functional
from torch.utils.data import Dataset
import torch
class ImageToTensor(Dataset):
    """
    A tensor dataset initiated with a PIL image dataset. Converts images to tensors when invoked. 

    Arguments: 
        image_dataset -- original PIL image dataset
    """
    def __init__(self, image_dataset, transform = transforms.ToTensor()):
        self.image_dataset = image_dataset
        self.transform = transform
    def __len__(self): 
        return len(self.image_dataset)
    
    #returns a tensor
    def __getitem__(self, idx): 
        return self.transform(self.image_dataset[idx])
            
class RotatedMNIST(Dataset): 
    """
    Rotated MNIST Dataset. Returns image, label, and rotation angle theta. 
    """
    def __init__(self, dataset, transform): 
        self.dataset = dataset
        self.transform = transform
        self.thetas = torch.rand(len(dataset)) * 2 * torch.pi
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx): 
        sample = self.dataset[idx]

        image = sample["image"]
        label = sample["label"]
        theta = self.thetas[idx]

        theta_deg = theta.item() * 180 / torch.pi

        image = transforms.functional.rotate(image, angle=theta_deg, fill=0)
        image = self.transform(image)

        return image, label, theta