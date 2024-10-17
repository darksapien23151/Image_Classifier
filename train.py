import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
from collections import OrderedDict

# Helper functions for saving the checkpoint and loading data
from utils import save_checkpoint, get_data_loaders

def train_model(args):
    # Load training, validation, and test datasets
    trainloader, validloader, testloader, class_to_idx = get_data_loaders(args.data_dir)
    
    # Choose the architecture (default is VGG16)
    model = models.vgg13(pretrained=True) if args.arch == 'vgg13' else models.vgg16(pretrained=True)

    # Freeze the pre-trained parameters so they don't get updated
    for param in model.parameters():
        param.requires_grad = False
    
    # Define the classifier to replace the model's original fully connected layers
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, args.hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(args.hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier

    # Move model to GPU if available and requested
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Start training
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass, compute the loss, backpropagate, and update weights
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print out training progress
            if steps % print_every == 0:
                model.eval()  # Turn off dropout for validation
                valid_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        valid_loss += criterion(logps, labels).item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()  # Switch back to training mode

    # Save the trained model checkpoint
    model.class_to_idx = class_to_idx
    save_checkpoint(model, args.save_dir, optimizer, epochs)

# Command-line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a new neural network on a dataset")
    parser.add_argument('data_dir', type=str, help='Directory containing the dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save the checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture: vgg13 or vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of units in the hidden layer')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train the model')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()
    train_model(args)

