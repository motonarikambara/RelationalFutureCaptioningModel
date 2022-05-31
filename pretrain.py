import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from tqdm import tqdm
import wandb
from torchvision import models
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch import nn
import copy
import pickle
import csv
import cv2


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None):
        if train == True:
            self.clip_file = "./annotations/ponnet/trainval.csv"
        else:
            self.clip_file = "./annotations/ponnet/test.csv"
        with open(self.clip_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            list_of_rows = list(csv_reader)
        self.csv_file = list_of_rows
        self.transform = transform


    def __len__(self):
        return len(self.csv_file)


    def __getitem__(self, i):
        raw_name = str(self.csv_file[i][0])
        frame_dir = "_" + raw_name
        file_n = os.path.join(".", "ponnet_data", "frames", frame_dir)
        feats = []
        for i in range(5):
            file_name = "frames_" + str(i) + ".png"
            img_path = os.path.join(file_n, file_name)

            image = cv2.imread(img_path)
            emb_feat = cv2.resize(image, (224, 224))
            feats.append(emb_feat)
        feats = np.array(feats).astype(np.float32)

        gt_file_n = os.path.join(".", "ponnet_data", "future_frames", frame_dir)
        gtfeats = []
        for i in range(5):
            file_name = "frames_" + str(i) + ".png"
            img_path = os.path.join(gt_file_n, file_name)

            image = cv2.imread(img_path)
            emb_feat = cv2.resize(image, (224, 224))
            gtfeats.append(emb_feat)
        gtfeats = np.array(gtfeats).astype(np.float32)
        return (feats, raw_name), gtfeats


class FeatureExtractor(torch.nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv1 = torch.nn.Conv3d(3, 256, (1, 8, 8), stride=(1, 2, 2))
        self.conv2 = torch.nn.Conv3d(256, 512, (1, 8, 8), stride=(1, 2, 2))
        self.conv3 = torch.nn.Conv3d(512, 1024, (1, 8, 8), stride=(1, 2, 2))
        self.conv4 = torch.nn.Conv3d(1024, 768, (1, 8, 8), stride=(1, 2, 2))
        self.conv5 = torch.nn.Conv3d(768, 512, (1, 8, 8), stride=(1, 1, 1))
        self.conv6 = torch.nn.Conv3d(512, 512, (1, 1, 1), stride=(1, 1, 1))
        self.conv7 = torch.nn.Conv3d(512, 512, (1, 1, 1), stride=(1, 1, 1))


    def forward(self, x): # input: (batch_size, 5, 224, 224, 3)
        x = x.permute(0, 4, 1, 2, 3)  # (batch_size, 3, 5, 224, 224)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x).reshape(-1, 5, 512)
        return x




def main():
    # wandb.init(name="l1norm", project="hossein_pretrain")

    trainset = MyDataset()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

    validset = MyDataset(train=False)
    validloader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=False)


    # net = FeatureExtractor()
    net = torch.load("./ponnet_data/model.pth")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    # optimizer = torch.optim.Adam(net.parameters())
    # criterion = nn.L1Loss()
    tmploss = 1e10

    dir_path = os.path.join(".", "ponnet_data", "frames_pkl")

    # for epoch in tqdm(range(30)):
    #     # 学習
    #     net.train()
    #     running_train_loss = 0.0
    #     with torch.set_grad_enabled(True):
    #         for data in tqdm(trainloader):
    #             (inputs, _), labels = data
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)
    #             optimizer.zero_grad()
    #             outputs = net(inputs)
    #             outputs_gt = net(labels)
    #             loss = criterion(outputs, outputs_gt)

    #             running_train_loss += loss.item()
    #             loss.backward()
    #             optimizer.step()

    #     wandb.log({"train_loss": running_train_loss})

        # 検証
        # net.eval()
        # running_valid_loss = 0.0

        # with torch.set_grad_enabled(False):
        #     for data in tqdm(validloader):
        #         (inputs, _), labels = data
        #         labels = labels.to(device)
        #         inputs = inputs.to(device)

        #         outputs = net(inputs)
        #         outputs_gt = net(labels)
        #         loss = criterion(outputs, outputs_gt)
        #         running_valid_loss += loss.item()

        # wandb.log({"valid_loss": running_valid_loss})
        # if running_valid_loss < tmploss:
        #     tmploss = running_valid_loss
        #     torch.save(net, "./ponnet_data/model.pth")

    net.eval()
    running_valid_loss = 0.0

    with torch.set_grad_enabled(False):
        for data in tqdm(trainloader):
            (inputs, raw_name), labels = data
            inputs = inputs.to(device)
            outputs = net(inputs)
            # print(raw_name[0])
            emb_path = os.path.join(dir_path, str(raw_name[0])+".pkl")
            with open(emb_path, "wb") as f:
                pickle.dump(outputs, f)
        for data in tqdm(validloader):
            (inputs, raw_name), labels = data
            inputs = inputs.to(device)
            outputs = net(inputs)
            emb_path = os.path.join(dir_path, str(raw_name[0])+".pkl")
            with open(emb_path, "wb") as f:
                pickle.dump(outputs, f)










if __name__ == "__main__":
    main()