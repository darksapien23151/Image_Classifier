import torch
from torchvision import datasets, transforms
from torch import nn
from PIL import Image
import numpy as np

def get_data_loaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return trainloader, validloader, testloader, train_data.class_to_idx

def save_checkpoint(model, save_dir, optimizer, epochs):
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state': optimizer.state_dict(),
        'epochs': epochs,
    }
    torch.save(checkpoint, save_dir)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(25088, 512),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(512, 102),
                                     nn.LogSoftmax(dim=1))
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a torch tensor.
    '''
    img = Image.open(image_path)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    img_tensor = preprocess(img)
    return img_tensor.unsqueeze(0)

def predict_image(image, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''
    image = image.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    
    return top_p.cpu().numpy()[0], top_class.cpu().numpy()[0]
