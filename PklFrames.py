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


class SubLayerT(nn.Module):
    def __init__(self):
        super(SubLayerT, self).__init__()

        self.conv1 = nn.Conv2d(3, 3, 4, stride=4) # (180, 320)
        self.resnet = models.resnet50(pretrained=True)
        self.fc1 = nn.Linear(512, 768)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x) # (b, 512, 23, 40)
        x = F.adaptive_avg_pool2d(x, (1, 1)) # (b, 512, 1, 1)
        x = torch.reshape(x,(-1, 512))
        # x = self.fc1(x)
        # x = torch.reshape(x,(-1, 1, 768))

        return x


def save_frames(video_path: str, frame_dir: str,
                name="frame", ext="pkl"):
    net = SubLayerT()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    v_name = splitext(basename(video_path))[0]
    if frame_dir[-1:] == "\\" or frame_dir[-1:] == "/":
        frame_dir = dirname(frame_dir)
    frame_dir_ = join(frame_dir, v_name)

    makedirs(frame_dir_, exist_ok=True)
    base_path = join(frame_dir_, name)

    idx = 0
    while cap.isOpened():
        idx += 1
        ret, frame = cap.read()
        if ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:  # 0秒のフレームを保存
                # cv2.imwrite("{}_{}.{}".format(base_path, "0", ext),
                #             frame)
                with open("{}_{}.{}".format(base_path, '0', ext), "wb") as f:
                    pickle.dump(frame, f)
            elif idx < cap.get(cv2.CAP_PROP_FPS):
                continue
            else:  # 1秒ずつフレームを保存
                second = int(cap.get(cv2.CAP_PROP_POS_FRAMES)/idx)
                if second > 4:
                    break
                # cv2.imwrite("{}_{}.{}".format(base_path, second, ext),
                #             frame)
                frame = torch.from_numpy(frame.astype(np.float32)).clone()
                frame = torch.reshape(frame, (1, 3, 720, 1280))
                frame = net(frame)
                frame = frame.to('cpu').detach().numpy().copy()
                with open("{}_{}.{}".format(base_path, second, ext), "wb") as f:
                    pickle.dump(frame, f)
                idx = 0
        else:
            break


if __name__ == "__main__":
    frame_dir = "./ponnet_data/frames_pkl/"
    clip_file = "./ponnet_data/1000samples.csv"

    with open(clip_file, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        header = next(reader)
        # reader = csv.reader(f)
        i = 0
        for row in tqdm(reader):
            file_name = '_' + row[0] + '.mp4'
            # dir_name = frame_dir + file_name.replace(".mov", "")
            save_frames("./ponnet_data/samples/" + file_name, frame_dir)
