import os
from turtle import st
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
import torchvision
import json
from tqdm import tqdm
import wandb
from torchvision import models
import torch.nn.functional as F
from torch import nn
import copy
import pickle
import re


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, label_path, transform=None):
        x = []
        y = []
        json_open = open(label_path, 'r')
        json_load = json.load(json_open)


        speed_std = 6.943259466752163
        acc_std = 1.0128755278649304
        crs_std = 105.43048660106768
        crs_vel_std = 23.557576723588763
        speed_mean = 6.592310560518758
        acc_mean = -0.032466484184198605
        crs_mean = 179.07880361238463
        crs_vel_mean = 0.09007327456722607


        for v in json_load:
            y_temp = []
            x.append(v["img_path"])
            # y_temp.append((float(v["speed"]) - speed_mean) / speed_std)
            y_temp.append((float(v["accelerater"]) - acc_mean) / acc_std)
            # y_temp.append((float(v["course"]) - crs_mean) / crs_std)
            # y_temp.append((float(v["course_vel"]) - crs_vel_mean) / crs_vel_std)
            y.append(y_temp)

        self.x = x
        # self.y = torch.from_numpy(np.array(y)).float().view(-1, 4, 1)
        self.y = torch.from_numpy(np.array(y)).float().view(-1, 1, 1)

        self.transform = transform


    def __len__(self):
        return len(self.x)


    def __getitem__(self, i):
        img = PIL.Image.open(self.x[i]).convert('RGB')

        img_dir = os.path.dirname(self.x[i])
        img_num = os.path.splitext(os.path.basename(self.x[i]))[0]
        img_num = img_num.replace('frame_', '')
        img_num = int(img_num)
        img_list = []
        img_list.append(img)
        img_trans_list = []
        for index in range(5):
            img_path = img_dir + "frame_" + str(img_num - index - 1) + ".png"
            is_file = os.path.isfile(img_path)
            if is_file:
                img_post = PIL.Image.open(img_path).convert('RGB')
                img_list.append(img_post)

        if self.transform is not None:
            for _img in img_list:
                _img = self.transform(_img)
                img_trans_list.append(_img)

        return img_trans_list, self.y[i]


class CNNLayer(nn.Module):
    def __init__(self):
        super(CNNLayer, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 2, stride=2) # (64, 64)
        self.conv2 = nn.Conv2d(16, 32, 2, stride=2) # (32, 32)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=4) # (8, 8)
        self.conv4 = nn.Conv2d(64, 128, 2, stride=2) # (4, 4)
        self.conv5 = nn.Conv2d(128, 1024, 4) # (1, 1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # x = x.view(-1, 4, 256)
        x = x.view(-1, 1, 1024)

        return x


class RegressionNet(nn.Module):

    def __init__(self):
        super(RegressionNet, self).__init__()

        layer = CNNLayer()
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(6)])

        # self.fc1 = nn.Linear(256 * 6, 768)
        # self.fc2 = nn.Linear(768, 192)
        # self.fc3 = nn.Linear(192, 64)
        # self.fc4 = nn.Linear(64, 8)
        # self.fc5 = nn.Linear(8, 1)

        self.fc1 = nn.Linear(1024 * 6, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 64)
        # self.fc4 = nn.Linear(64, 16)
        # self.fc5 = nn.Linear(16, 4)
        # self.fc5 = nn.Linear(4, 1)


    def forward(self, x_list):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        index = 0
        feature_img_list = []
        for _x in x_list:
            _x = self.layer[index](_x)
            feature_img_list.append(_x)
            index += 1
        x = torch.cat(feature_img_list, dim=2)

        # (64or16, 4, 256*6)にしたい
        zero_size = 1024 * 6 - x.size()[2]
        zeros = torch.zeros(x.size()[0], x.size()[1], zero_size)
        zeros = zeros.to(device)
        x = torch.cat([x, zeros], dim=2)

        # # (64or16, 1, 1024*6)にしたい
        # zero_size = 1024 * 6 - x.size()[2]
        # zeros = torch.zeros(x.size()[0], x.size()[1], zero_size)
        # zeros = zeros.to(device)
        # x = torch.cat([x, zeros], dim=2)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # x = self.fc4(x)
        # x = self.fc5(x)

        return x


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()

        self.layer = RegressionNet()

        self.fc4 = nn.Linear(64, 8)
        self.fc5 = nn.Linear(8, 1)


    def forward(self, x_list):
        x = self.layer(x_list)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        return x


def main():
    wandb.init(name="pre-sensor", project="BDD-X")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    train_data_dir = './annotations/BDD-X/bddx_sensor_train.json'
    valid_data_dir = './annotations/BDD-X/bddx_sensor_valid.json'
    test_data_dir = './annotations/BDD-X/bddx_sensor_test.json'

    trainset = MyDataset(train_data_dir, transform=transform)
    trainset.x = trainset.x[:10000]
    trainset.y = trainset.y[:10000]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    validset = MyDataset(valid_data_dir, transform=transform)
    validset.x = validset.x[:2000]
    validset.y = validset.y[:2000]
    validloader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=False)

    train_image_id = './annotations/BDD-X/captioning_train.json'
    train_id_path = './annotations/BDD-X/bddx_id_train.json'

    valid_image_id = './annotations/BDD-X/captioning_val.json'
    valid_id_path = './annotations/BDD-X/bddx_id_valid.json'

    test_image_id = './annotations/BDD-X/captioning_test.json'
    test_id_path = './annotations/BDD-X/bddx_id_test.json'

    # valid_rfcmDataset = RFCMDataset(valid_data_dir, valid_image_id, valid_id_path, transform)
    # valid_rfcmDataloader = torch.utils.data.DataLoader(valid_rfcmDataset, batch_size=1, shuffle=True)

    # test_rfcmDataset = RFCMDataset(test_data_dir, test_image_id, test_id_path, transform)
    # test_rfcmDataloader = torch.utils.data.DataLoader(test_rfcmDataset, batch_size=1, shuffle=True)

    # train_rfcmDataset = RFCMDataset(train_data_dir, train_image_id, train_id_path, transform)
    # train_rfcmDataloader = torch.utils.data.DataLoader(train_rfcmDataset, batch_size=1, shuffle=True)


    net = MainNet()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters())
    criterion = nn.MSELoss()

    valid_out = []

    for epoch in tqdm(range(30)):
        # 学習
        net.train()
        running_train_loss = 0.0
        with torch.set_grad_enabled(True):
            for data in tqdm(trainloader):
                inputs, labels = data
                input_list = []
                for _inputs in inputs:
                    _inputs = _inputs.to(device)
                    input_list.append(_inputs)
                # inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = net(input_list)

                # vel, acc, crs, crs_velでMSE
                # loss_vel = criterion(outputs[:, 0, :], labels[:, 0, :])
                loss_acc = criterion(outputs[:, 0, :], labels[:, 0, :])
                # loss_crs = criterion(outputs[:, 2, :], labels[:, 2, :])
                # loss_crs_vel = criterion(outputs[:, 3, :], labels[:, 3, :])

                # lam_vel = 1 / 120
                # lam_acc = 1 / 4
                # lam_crs = 1 / 50000
                # lam_crs_vel = 1 / 6000

                # loss = loss_vel + loss_acc + loss_crs + loss_crs_vel
                loss = loss_acc

                # loss = lam_vel * loss_vel + lam_acc * loss_acc + lam_crs * loss_crs + lam_crs_vel * loss_crs_vel

                # loss = criterion(outputs, labels)
                running_train_loss += loss.item()
                loss.backward()
                optimizer.step()
                # wandb.log({"tloss_vel": loss_vel,
                #            "tloss_acc": loss_acc,
                #            "tloss_crs": loss_crs,
                #            "tloss_crs_vel": loss_crs_vel})
                wandb.log({"tloss_acc": loss_acc})

        wandb.log({"train_loss": running_train_loss})

        # 検証
        net.eval()
        running_valid_loss = 0.0
        val_vel = val_acc = val_crs = val_crs_vel = 0.0

        with torch.set_grad_enabled(False):
            for data in tqdm(validloader):
                inputs, labels = data
                input_list = []
                for _inputs in inputs:
                    _inputs = _inputs.to(device)
                    input_list.append(_inputs)
                # inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(input_list)

                # 出力
                for index in range(len(outputs[:, 0, :].tolist())):
                    # valid_out.append({"o_vel": outputs[:, 0, :].tolist()[index][0],
                    #                 "o_acc": outputs[:, 1, :].tolist()[index][0],
                    #                 "o_crs": outputs[:, 2, :].tolist()[index][0],
                    #                 "o_crs_vel" : outputs[:, 3, :].tolist()[index][0],
                    #                 "l_vel": labels[:, 0, :].tolist()[index][0],
                    #                 "l_acc": labels[:, 1, :].tolist()[index][0],
                    #                 "l_crs": labels[:, 2, :].tolist()[index][0],
                    #                 "l_crs_vel" : labels[:, 3, :].tolist()[index][0],
                    #                 })
                    valid_out.append({"o_acc": outputs[:, 0, :].tolist()[index][0],
                                    "l_acc": labels[:, 0, :].tolist()[index][0],
                                    })

                # vel, acc, crs, crs_velでMSE
                val_acc += criterion(outputs[:, 0, :], labels[:, 0, :]) / len(validset)
                # val_acc += criterion(outputs[:, 1, :], labels[:, 1, :]) / len(validset)
                # val_crs += criterion(outputs[:, 2, :], labels[:, 2, :]) / len(validset)
                # val_crs_vel += criterion(outputs[:, 3, :], labels[:, 3, :]) / len(validset)

            # loss = loss_vel + loss_acc + loss_crs + loss_crs_vel
            loss = val_acc
            running_valid_loss += loss

            # wandb.log({"valid_loss": running_valid_loss,
            #        "valid_vel": val_vel,
            #        "valid_acc": val_acc,
            #        "valid_crs": val_crs,
            #        "valid_crs_vel": val_crs_vel})
            wandb.log({"valid_loss": running_valid_loss,
                   "valid_acc": val_acc,})

        # print('#epoch:{}  train loss: {}  valid loss: {}  valid vel: {}  valid acc: {}  valid crs: {}  valid crs vel: {}'.format(epoch,
        #                                                                                                             running_train_loss,
        #                                                                                                             running_valid_loss,
        #                                                                                                             val_vel,
        #                                                                                                             val_acc,
        #                                                                                                             val_crs,
        #                                                                                                             val_crs_vel))
        print('#epoch:{}  train loss: {}  valid loss: {}  valid acc: {}'.format(epoch,
                                                                        running_train_loss,
                                                                        running_valid_loss,
                                                                        val_acc,))


    # outputs_reg = net.layer(input_list)


if __name__ == "__main__":
    main()
