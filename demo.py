import os
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
from nlfd import build_model


def demo(model_path, img_path, cuda):
    transform = transforms.Compose([transforms.Resize((352, 352)), transforms.ToTensor()])
    img = Image.open(img_path)
    shape = img.size
    img = transform(img) - torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1) / 255
    img = Variable(img.unsqueeze(0), volatile=True)
    net = build_model()
    net.load_state_dict(torch.load(model_path))
    net.eval()
    if cuda: img, net = img.cuda(), net.cuda()
    prob = net(img)
    prob = (prob.cpu().data[0][0].numpy() * 255).astype(np.uint8)
    p_img = Image.fromarray(prob, mode='L').resize(shape)
    p_img.show()


if __name__ == '__main__':
    model_path = './weights/best.pth'
    img_path = './png/demo.jpg'
    parser = argparse.ArgumentParser()

    parser.add_argument('--demo_img', type=str, default=img_path)
    parser.add_argument('--trained_model', type=str, default=model_path)
    parser.add_argument('--cuda', type=bool, default=True)
    config = parser.parse_args()
    ext = ['.jpg', '.png']
    if not os.path.splitext(config.demo_img)[-1] in ext:
        raise IOError('illegal image path')

    demo(config.trained_model, config.demo_img, config.cuda)
