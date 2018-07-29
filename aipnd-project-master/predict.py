import argparse
import json

from PIL import Image
import torch
from torch.autograd import Variable
import numpy as np

def arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input', help = 'The model input',
                        type = str)
    parser.add_argument('trained_model', help = 'Use the model that has been trained',
                        type = str)
    parser.add_argument('-t', '--topk', type = int, default = 5,
                        dest = 'topk',
                        help = 'Top model output possibilities')
    parser.add_argument('-c', '--category_names', type = str, default = 'cat_to_name.json',
                        dest = 'cat_names',
                        help = 'File contain data id and names')
    parser.add_argument('-g', '--gpu', default = False, action = 'store_true',
                        dest = 'gpu',
                        help = 'GPU mode')
    return parser.parse_args()

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    ratio = image.size[1] / image.size[0]
    image = image.resize((256, int(ratio * 256)))
    half_width = image.size[0] / 2
    half_height = image.size[1] / 2
    
    cropped_image = image.crop(
        (
        half_width - 112,
        half_height - 112,
        half_width + 112,
        half_height + 112
        )
    )
    image_np = np.array(cropped_image) / 225.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np - mean) / std
    image_np = image_np.transpose((2, 0, 1))
    
    return image_np

def load_checkpoint(filepath):
    Load = torch.load(filepath)
    epochs = Load['epochs']
    optimizer = Load['optimizer_dict']
    model = Load['model']
    model.class_to_idx = Load['class_to_idx']
    model.classifier = Load['classifier']
    model.load_state_dict(Load['state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    return model

def predict(image_path, model, topk, gpu = False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    device = 'cuda:0' if torch.cuda.is_available else 'cpu'
    if torch.cuda.is_available() & gpu == True:
        model = model.cuda()
    with torch.no_grad():
        model.eval()
        img_np = process_image(image_path)
        img_tensor = torch.from_numpy(img_np)
        
        if torch.cuda.is_available() & gpu == True:
            inputs_var = Variable(img_tensor.float().cuda())
        else:
            inputs_var = Variable(img_tensor.float())
        inputs_var = Variable(img_tensor.float().cuda())    
        
        output = model.forward(inputs_var.unsqueeze(0))
        ps = torch.exp(output).topk(topk)
        probs = ps[0].cuda() if torch.cuda.is_available & gpu == True else ps[0]
        classes = ps[1].cuda() if torch.cuda.is_available & gpu == True else ps[1]
        mapped_class = []
        for i in classes.numpy()[0]:
            mapped_class.append(model.class_to_idx[str(i)])
        name_list = []
        with open('cat_to_name.json') as file:
            data = json.load(file)
            for flower_id in classes:
                name_list.append(data[str(flower_id)])
    return probs.numpy()[0], name_list


if __name__=='__main__':
    args = arguments()
    model = load_checkpoint(args.trained_model)
    print(type(args.gpu))
    probs, names = predict(args.input, model, args.topk, args.gpu)
    print('{}\n{}'.format(names, probs))