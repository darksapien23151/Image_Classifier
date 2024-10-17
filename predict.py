import torch
import json
from torch import nn
from torchvision import models
from PIL import Image
import argparse

# Helper functions to load the model, process images, and make predictions
from utils import load_checkpoint, process_image, predict_image

def predict(args):
    # Load the model from the saved checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # Process the input image
    image = process_image(args.input)
    
    # Use GPU if available and requested
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Make the prediction
    probs, classes = predict_image(image, model, args.top_k, device)
    
    # Map class indices to actual flower names if a category names file is provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        names = [cat_to_name[str(cl)] for cl in classes]
    else:
        names = classes

    # Print out the top K predictions
    for prob, name in zip(probs, names):
        print(f"{name}: {prob*100:.2f}%")

# Command-line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict the flower class from an image using a trained model")
    parser.add_argument('input', type=str, help='Path to the image to predict')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='JSON file to map category indices to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available for prediction')

    args = parser.parse_args()
    predict(args)

