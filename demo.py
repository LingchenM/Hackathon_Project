import torch
import torchvision
from PIL import Image
from torch.nn import Module
from random import randint

number = randint(0, 9)
index_1 = randint(1, 2)
index_2 = randint(0, 199)


image = Image.open(f'D:\Desktop\Hack Project\hackathon\credit_card\credit_card_original\\test\\{number}\_{index_1}_{index_2}.jpg')

trans = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])

image = trans(image)

model = torch.load('D:\Desktop\Hack Project\hackathon\checkpoints\\result9.pth', map_location='cpu')

image = torch.reshape(image, (1, 1, 32, 32))
model.eval()

with torch.no_grad():
    output = model(image)

target_ls = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

print(f'Actual image is {number}')
print(f'Image recognized as {target_ls[output.argmax(1).item()]}')