import numpy as np
import torch
import PIL
from model import Shape_Effi_B7, Shape_Effi_B0, Shape_ResNet152, Shape_ResNet18
import torchvision.transforms as transform

model = Shape_Effi_B0(2)
model.load_state_dict(torch.load('shape_model.pt'))

sample = PIL.Image.open('test/test5.jpg')
tf = transform.Compose([transform.Grayscale(3), transform.Resize((224, 224)), transform.ToTensor()])
sample = tf(sample).unsqueeze(0)

output = model(sample)
output_idx = torch.argmax(output).numpy()
sm = torch.nn.functional.softmax(output, dim=1)
label = ['원', '타원']
print('{} {:.2f}%'.format(label[output_idx], max(sm[0].data.numpy())*100))