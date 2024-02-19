import argparse
import json
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms 
import torchvision.models as models

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar='image_path', type=str, default='flowers/test/25/image_06580.jpg')
    parser.add_argument('checkpoint', metavar='checkpoint', type=str, default='train_checkpoint.pth')
    parser.add_argument('--top_k', action='store', dest="top_k", type=int, default=5)
    parser.add_argument('--category_names', action='store', dest='category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=False)
    return parser.parse_args()

# def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
#     model = models.__dict__[checkpoint['arch']](pretrained=True)
#     model.classifier = checkpoint['classifier']
#     model.load_state_dict(checkpoint['state_dict'])
#     model.class_to_idx = checkpoint['class_to_idx']
#     return model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

    pretrained_model = checkpoint.get('pretrained_model', 'vgg16')
    
    # Use weights instead of pretrained, and provide the correct enum
    model = getattr(models, pretrained_model)(weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    model.input_size = checkpoint['input_size']
    model.output_size = checkpoint['output_size']
    model.learning_rate = checkpoint['learning_rate']
    
    # Check if 'hidden_units' is present in the checkpoint
    model.hidden_units = checkpoint.get('hidden_units', 4096)  # Default to 4096 if not present
    
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint['optimizer']
    
    return model



# def process_image(image_path):
#     image = Image.open(image_path)
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     image = transform(image)
#     return image

def process_image(pil_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Resize
    pil_image = pil_image.resize((256, 256))
    
    # Center crop
    width, height = pil_image.size
    new_width, new_height = 224, 224
    
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = left + new_width
    bottom = top + new_height
    
    pil_image = pil_image.crop((left, top, right, bottom))
    
    # Convert color channels from 0-255 to 0-1
    np_image = np.array(pil_image) / 255
    
    # Normalize for model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose color channels to 1st dimension
    np_image = np_image.transpose((2, 0, 1))
    
    # Convert to Float Tensor
    tensor = torch.from_numpy(np_image)
    tensor = tensor.type(torch.FloatTensor)
   
    # Return tensor
    return tensor

def predict(image_path, model, top_k, gpu):
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    image = Image.open(image_path)
    image = process_image(image)  # This already returns a PyTorch tensor
    image = image.unsqueeze(0).float()

    with torch.no_grad():
        output = model.forward(image.to(device))

    p = F.softmax(output.data, dim=1)

    top_p = np.array(p.topk(top_k)[0][0])

    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    # top_classes = [np.int(index_to_class[each]) for each in np.array(p.topk(top_k)[1][0])]
    top_classes = [int(index_to_class[each]) for each in np.array(p.topk(top_k)[1][0])]

    return top_p, top_classes, device

def load_names(category_names_file):
    with open(category_names_file) as file:
        category_names = json.load(file)
    return category_names

gpu = False

def main():
    args = parse_args()
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu

    model = load_checkpoint(checkpoint)

    top_p, classes, device = predict(image_path, model, top_k, gpu)

    category_names = load_names(category_names)

    labels = [category_names[str(index)] for index in classes]

    print(f"Results for your File: {image_path}")
    print(labels)
    print(top_p)
    print()

    for i in range(len(labels)):
        print("{} - {} with a probability of {}".format((i+1), labels[i], top_p[i]))

if __name__ == "__main__":
    main()
