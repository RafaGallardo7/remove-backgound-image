from flask import Flask
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

app = Flask(__name__)

def load_model():
  model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
  model.eval()
  return model

def make_transparent_foreground(pic, mask):  
  b, g, r = cv2.split(np.array(pic).astype('uint8'))  
  a = np.ones(mask.shape, dtype='uint8') * 255  
  alpha_im = cv2.merge([b, g, r, a], 4)  
  bg = np.zeros(alpha_im.shape)  
  new_mask = np.stack([mask, mask, mask, mask], axis=2)  
  foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

  return foreground

def remove_background(model, input_file):
  input_image = Image.open(input_file)
  new_height = 500
  new_width  = new_height * input_image.width / input_image.height
  new_size = (int(new_width), int(new_height))
  resized_image = input_image.resize(new_size)  
  
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  input_tensor = preprocess(resized_image)
  input_batch = input_tensor.unsqueeze(0)
  
  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')

  with torch.no_grad():
      output = model(input_batch)['out'][0]
  output_predictions = output.argmax(0)

  mask = output_predictions.byte().cpu().numpy()
  background = np.zeros(mask.shape)
  bin_mask = np.where(mask, 255, background).astype(np.uint8)

  foreground = make_transparent_foreground(resized_image ,bin_mask)

  return foreground, bin_mask

deeplab_model = load_model()
foreground, bin_mask = remove_background(deeplab_model, 'img.jpg')
plt.imshow(foreground)
plt.axis('off')
plt.show()
new_image = Image.fromarray(foreground)
new_image.save('result.png')


if __name__ == '__main__':
    app.run(debug=True)
