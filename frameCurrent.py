import cv2
from os import makedirs
from os.path import splitext, dirname, basename, join
import csv
import subprocess
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import pickle
from torch import nn
from PIL import Image


class SubLayerT(nn.Module):
    def __init__(self):
        super(SubLayerT, self).__init__()

        # self.conv1 = nn.Conv2d(3, 3, 4, stride=4) # (180, 320)
        self.resnet = models.resnet50(pretrained=True)
        # self.fc1 = nn.Linear(512, 768)

    def forward(self, x):
        # x = self.conv1(x)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        # print(x.shape)
        # x = self.resnet.layer2(x) # (b, 512, 23, 40)
        x = F.adaptive_avg_pool2d(x, (2, 2)) # (b, 256, 2, 2)
        x = torch.reshape(x,(-1, 1024))
        # x = self.fc1(x)
        # x = torch.reshape(x,(-1, 1, 768))

        return x


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2 + 80,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2 + 80))


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image



def save_frames(video_path: str, frame_dir: str,
                name="frame", ext="pkl", file_name="tmp"):
    net = SubLayerT()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    # v_name = splitext(basename(video_path))[0]
    # if frame_dir[-1:] == "\\" or frame_dir[-1:] == "/":
    #     frame_dir = dirname(frame_dir)
    # frame_dir_ = join(frame_dir, v_name)

    # makedirs(frame_dir_, exist_ok=True)
    base_path = frame_dir

    idx = 0
    while cap.isOpened():
        idx += 1
        ret, frame = cap.read()
        if ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:  # 0秒のフレームを保存
                # cv2.imwrite("{}_{}.{}".format(base_path, "0", ext),
                #             frame)
                # with open("{}_{}.{}".format(base_path, '0', ext), "wb") as f:
                #     pickle.dump(frame, f)
                pass
            elif idx < cap.get(cv2.CAP_PROP_FPS):
                continue
            else:  # 1秒ずつフレームを保存
                second = int(cap.get(cv2.CAP_PROP_POS_FRAMES)/idx)
                if second > 1:
                    break
                # cv2.imwrite("{}_{}.{}".format(base_path, second, ext),
                #             frame)
                # 切り抜き
                frame = cv2.resize(frame, dsize=(500, 500))
                frame = cv2pil(frame)
                frame = crop_center(frame, 224, 224)
                frame = pil2cv(frame)
                # frame = cv2.resize(frame, dsize=(64, 64))

                frame = torch.from_numpy(frame.astype(np.float32)).clone()
                frame = torch.reshape(frame, (1, 3, 224, 224))
                frame = net(frame)
                # frame = frame.reshape((64, 64, 3))
                frame = frame.to('cpu').detach().numpy().copy()
                # cv2.imwrite("./gt.png", frame)
                with open(join(base_path, "{}.{}".format(file_name, ext)), "wb") as f:
                    pickle.dump(frame, f)
                idx = 0
        else:
            break


if __name__ == "__main__":
    frame_dir = "./ponnet_data/res_frames_pkl/"
    clip_file = "./ponnet_data/1000samples.csv"

    with open(clip_file, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        # header = next(reader)
        # reader = csv.reader(f)
        i = 0
        for row in tqdm(reader):
            file_name = '_' + row[0] + '.mp4'
            # dir_name = frame_dir + file_name.replace(".mov", "")
            save_frames("./ponnet_data/samples/" + file_name, frame_dir, file_name=row[0])