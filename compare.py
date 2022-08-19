import torch
from PIL import Image
from torchvision import transforms
import numpy as np

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
model.eval()
if torch.cuda.is_available():
    model.to('cuda')

import megengine as mge
import megengine.functional.nn as nn
from resnextt_mge import resnext50_32x4d

checkpoint_mge = mge.hub.load_serialized_obj_from_url(
    'https://studio.brainpp.com/api/v1/activities/3/missions/106/files/d1e3089c-3113-4392-a2bb-ebca4183ce8a')
# checkpoint_mge = mge.load('./resnext50_32x4d.mge')
model_mge = resnext50_32x4d()
model_mge.eval()
model_mge.load_state_dict(checkpoint_mge)





def get_data(filename):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
    return input_batch

import os
image_folder = './data'
for i, image_name in enumerate(os.listdir(image_folder)):
    image_path = os.path.join(image_folder, image_name)
    input_batch = get_data(image_path)
    # torch model inference
    with torch.no_grad():
        output = model(input_batch)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    output_pt = np.argmax(probabilities.detach().cpu().numpy())
    # megengine model inference
    input_tensor_mge = mge.Tensor(input_batch.detach().cpu().numpy())
    output_mge = model_mge(input_tensor_mge)
    probabilities_mge = nn.softmax(output_mge[0], axis=0)
    output_mge = np.argmax(probabilities_mge.numpy())
    # differerce
    np.testing.assert_allclose(output_pt, output_mge, rtol=1e-3)
    print('The {}th image named {} passed'.format(i, image_name))

print('Counting inference time......')

# time compare
iterations = 50
import time
image_path = './data/dog.jpg'
input_batch = get_data(image_path)
input_tensor_mge = mge.Tensor(input_batch.detach().cpu().numpy())

# torch model inference time
time_start = time.time()  # 记录开始时间
for iter in range(iterations):
    # torch model inference
    with torch.no_grad():
        _ = model(input_batch)
time_end = time.time()  # 记录结束时间
mean_time_pt = (time_end - time_start)/iterations
print(" Torch model inference time: {:.6f}s, FPS: {} ".format(mean_time_pt, 1/mean_time_pt))


# mge model inference time
time_start = time.time()  # 记录开始时间
for iter in range(iterations):
    _ = model_mge(input_tensor_mge)
time_end = time.time()  # 记录结束时间
mean_time_mge = (time_end - time_start)/iterations
print(" Megengine model inference time: {:.6f}s, FPS: {} ".format(mean_time_mge, 1/mean_time_mge))





# Read the categories
# with open("imagenet_classes.txt", "r") as f:
#     categories = [s.strip() for s in f.readlines()]
#
# # Show top categories per image
# top5_prob, top5_catid = torch.topk(probabilities, 5)
#
# for i in range(top5_prob.size(0)):
#     print(categories[top5_catid[i]], top5_prob[i].item())
