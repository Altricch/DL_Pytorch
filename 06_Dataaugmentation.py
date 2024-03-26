import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from Customdataset import CatsAndDogsDataset


# Load Data
my_transforms = transforms.Compose([
    # Transformations work on this format
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees = 0.45),
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.0,0.0,0.0], std=[1.0, 1.0, 1.0]) # Usually find values first for each channel and then input it
])

dataset = CatsAndDogsDataset(csv_file = "cats_dogs.csv", root_dir= "cats_dogs_resized", transform = my_transforms)

for img_num, (img, label) in enumerate(dataset):
    save_image(img, "cats_dogs_transformed/catdogstrans"+str(img_num)+".png")