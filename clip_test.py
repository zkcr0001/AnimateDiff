from sentence_transformers import SentenceTransformer
import numpy as np
import requests
from PIL import Image
from animatediff.data.video_dataset import VideoDataset, collate_fn
import torch

clip = SentenceTransformer('clip-ViT-L-14').cuda()
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).resize((192, 128))
image = [image] * 2 # (B, PIL image)

print("--------Image--------")
for i in range(10):
    f = clip.encode(image)
    print(f.shape)


train_dataset = VideoDataset(["/home/ubuntu/Pose_dataset/Processed_video/Fashion_test"], sample_size=(192, 128))
train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, num_workers=0,)

print("--------Video Dataset--------")
for idx, batch in enumerate(train_dataloader):

    if idx > 3:
        break

    first_image = [batch["images"][b][0] for b in range(len(batch["images"]))]
    f = clip.encode(first_image)

    print(first_image)
    print(f.shape)