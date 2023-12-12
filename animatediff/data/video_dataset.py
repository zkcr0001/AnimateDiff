import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from decord import VideoReader


class VideoDataset(Dataset):
    def __init__(self, folder_list, sample_stride=None, sample_frame=16, sample_size=(512, 384), only_train_image=False, transform=None):
        self.folder_list = folder_list
        self.transform = transform
        self.only_train_image = only_train_image
        self.sample_stride = sample_stride
        self.sample_frame = sample_frame
        self.sample_size = sample_size

        assert(sample_size[0] % 64 == 0 and sample_size[1] % 64 == 0)

        video_folder_list = []
        data_list = []
        for folder in sorted(folder_list):
            folder = Path(folder)
            
            for video in sorted(os.listdir(folder)):
                video_folder = folder / video
                video_folder_list.append(video_folder)
                image_video_path = video_folder / "images.mp4"
                openpose_video_path = video_folder / "openpose.mp4"

                data_list.append((image_video_path, openpose_video_path))
                
        self.video_folder_list = video_folder_list
        self.data_list = data_list

        self.norm_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        

    def __getitem__(self, index):
        
        video_folder = self.video_folder_list[index]
        is_tictok = False
        if "Tic-Tok" in str(video_folder):
            is_tictok = True

        image_video_path, openpose_video_path = self.data_list[index]

        image_reader = VideoReader(str(image_video_path))
        openpose_reader = VideoReader(str(openpose_video_path))

        video_length = len(openpose_reader)
        temp_img = image_reader.get_batch([0]).asnumpy()

        _, ori_H, ori_W, _ = temp_img.shape

        H = ori_H
        W = ori_W
        if is_tictok:
            crop_H = ori_H - int(ori_H * 0.93)
            H = ori_H - 2 * crop_H
        
        # Dimension of the input image should be divisible by 64
        resize_H = self.sample_size[0] + 32
        resize_W = int(resize_H / H * W)
        new_W = self.sample_size[1]
        new_H = self.sample_size[0]

        # Randomly sample stride and clip length
        sample_stride = torch.randint(1, 7, (1,)).item() if self.sample_stride is None else self.sample_stride
        clip_length = min(video_length, (self.sample_frame - 1) * sample_stride + 1) if not self.only_train_image else 1

        # Randomly sample start frame and decide the batch index
        start_frame = torch.randint(0, video_length - clip_length + 1, (1,)).item()
        batch_index = np.linspace(start_frame, start_frame + clip_length - 1, self.sample_frame, dtype=int) if not self.only_train_image else [start_frame]

        # Read images and openposes
        images_out = torch.from_numpy(image_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        images_out = images_out / 255.
        openposes_out = torch.from_numpy(openpose_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        openposes_out = openposes_out / 255.
        openposes_out = transforms.Resize((ori_H, ori_W))(openposes_out)

        del image_reader, openpose_reader

        # Delete Tictok's top logo
        if is_tictok:
            images_out = images_out[:, :, crop_H:-crop_H, :]
            openposes_out = openposes_out[:, :, crop_H:-crop_H, :]
        images_out = transforms.Resize((resize_H, resize_W))(images_out)
        openposes_out = transforms.Resize((resize_H, resize_W))(openposes_out)

        # Crop or pad the image to the desired size
        if resize_W < new_W:
            left_W = torch.randint(0, new_W - resize_W + 1, (1,)).item()
            left_H = torch.randint(0, resize_H - new_H + 1, (1,)).item() // 2
            images_out = torch.nn.functional.pad(images_out, (left_W, new_W - resize_W - left_W, 0, 0), mode='constant', value=1)
            openposes_out = torch.nn.functional.pad(openposes_out, (left_W, new_W - resize_W - left_W, 0, 0), mode='constant', value=0)
            images_out = images_out[:, :, left_H:left_H + new_H, :]
            openposes_out = openposes_out[:, :, left_H:left_H + new_H, :]
        else:
            left_W = torch.randint(0, resize_W - new_W + 1, (1,)).item()
            left_H = torch.randint(0, resize_H - new_H + 1, (1,)).item() // 2
            images_out = images_out[:, :, left_H:left_H + new_H, left_W:left_W + new_W]
            openposes_out = openposes_out[:, :, left_H:left_H + new_H, left_W:left_W + new_W]

        # Randomly flip the image
        if torch.rand(1) > 0.5:
            images_out = torch.flip(images_out, dims=(3,))
            openposes_out = torch.flip(openposes_out, dims=(3,))

        # Normalize the image
        images_out = self.norm_transforms(images_out)

        return dict(pixel_values=images_out, openpose_values=openposes_out)
    

    def __len__(self):
        return len(self.data_list)
    

if __name__ == "__main__":

    folder_list = [Path("/home/ubuntu/Pose_dataset/Processed_video/Fashion_test"), Path("/home/ubuntu/Pose_dataset/Processed_video/Tic-Tok_test")]

    print("--------Video Dataset--------")
    dataset = VideoDataset(folder_list)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, batch["openpose_values"].shape)
        if idx == 3:
            break

    print("--------Image Dataset--------")
    dataset = VideoDataset(folder_list, only_train_image=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, batch["openpose_values"].shape)
        if idx == 3:
            break