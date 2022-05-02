import os
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


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, label_path, transform=None):
        x = []
        y = []
        json_open = open(label_path, 'r')
        json_load = json.load(json_open)

        for v in json_load:
            y_temp = []
            x.append(v["img_path"])
            y_temp.append(float(v["speed"]))
            y_temp.append(float(v["accelerater"]))
            y_temp.append(float(v["course"]))
            y_temp.append(float(v["course_vel"]))
            y.append(y_temp)

        self.x = x
        self.y = torch.from_numpy(np.array(y)).float().view(-1, 4, 1)

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
            img_path = img_dir + str(img_num - index - 1) + ".png"
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

        x = x.view(-1, 4, 256)
        # x = x.view(-1, 1, 1024)

        return x


class RegressionNet(nn.Module):

    def __init__(self):
        super(RegressionNet, self).__init__()

        layer = CNNLayer()
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(6)])

        self.fc1 = nn.Linear(256 * 6, 768)
        self.fc2 = nn.Linear(768, 192)
        self.fc3 = nn.Linear(192, 64)
        self.fc4 = nn.Linear(64, 8)
        self.fc5 = nn.Linear(8, 1)

        # self.fc1 = nn.Linear(1024 * 6, 2048)
        # self.fc2 = nn.Linear(2048, 512)
        # self.fc3 = nn.Linear(512, 64)
        # self.fc4 = nn.Linear(64, 16)
        # self.fc5 = nn.Linear(16, 4)


    def forward(self, x_list, epoch, data_num):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        index = 0
        feature_img_list = []
        for _x in x_list:
            _x = self.layer[index](_x)
            feature_img_list.append(_x)
            index += 1
        x = torch.cat(feature_img_list, dim=2)

        # (64or16, 4, 256*6)にしたい
        zero_size = 256 * 6 - x.size()[2]
        zeros = torch.zeros(x.size()[0], x.size()[1], zero_size)
        zeros = zeros.to(device)
        x = torch.cat([x, zeros], dim=2)

        # # (64or16, 1, 1024*6)にしたい
        # zero_size = 1024 * 6 - x.size()[2]
        # zeros = torch.zeros(x.size()[0], x.size()[1], zero_size)
        # zeros = zeros.to(device)
        # x = torch.cat([x, zeros], dim=2)

        x = self.fc1(x)
        x = self.fc2(x)
        if epoch == 29 and data_num == 156:
            with open("./out/img_feature.pkl", mode="wb") as f:
                pickle.dump(x, f)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)

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

    trainset = MyDataset(train_data_dir, transform=transform)
    trainset.x = trainset.x[:10000]
    trainset.y = trainset.y[:10000]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    validset = MyDataset(valid_data_dir, transform=transform)
    validset.x = validset.x[:2000]
    validset.y = validset.y[:2000]
    validloader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=False)

    net = RegressionNet()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters())
    criterion = nn.MSELoss()

    valid_out = []

    data_num = 0


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
                outputs = net(input_list, epoch, data_num)
                data_num += 1

                # vel, acc, crs, crs_velでMSE
                loss_vel = criterion(outputs[:, 0, :], labels[:, 0, :])
                loss_acc = criterion(outputs[:, 1, :], labels[:, 1, :])
                loss_crs = criterion(outputs[:, 2, :], labels[:, 2, :])
                loss_crs_vel = criterion(outputs[:, 3, :], labels[:, 3, :])

                lam_vel = 1 / 120
                lam_acc = 1 / 4
                lam_crs = 1 / 50000
                lam_crs_vel = 1 / 6000

                loss = lam_vel * loss_vel + lam_acc * loss_acc + lam_crs * loss_crs + lam_crs_vel * loss_crs_vel

                # loss = criterion(outputs, labels)
                running_train_loss += loss.item()
                loss.backward()
                optimizer.step()
                wandb.log({"loss_vel": loss_vel,
                           "loss_acc": loss_acc,
                           "loss_crs": loss_crs,
                           "loss_crs_vel": loss_crs_vel})
            data_num = 0

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
                outputs = net(input_list, 0, 0)

                # 出力
                for index in range(len(outputs[:, 0, :].tolist())):
                    valid_out.append({"o_vel": outputs[:, 0, :].tolist()[index][0],
                                    "o_acc": outputs[:, 1, :].tolist()[index][0],
                                    "o_crs": outputs[:, 2, :].tolist()[index][0],
                                    "o_crs_vel" : outputs[:, 3, :].tolist()[index][0],
                                    "l_vel": labels[:, 0, :].tolist()[index][0],
                                    "l_acc": labels[:, 1, :].tolist()[index][0],
                                    "l_crs": labels[:, 2, :].tolist()[index][0],
                                    "l_crs_vel" : labels[:, 3, :].tolist()[index][0],
                                    })

                # vel, acc, crs, crs_velでMSE
                val_vel += criterion(outputs[:, 0, :], labels[:, 0, :])
                val_acc += criterion(outputs[:, 1, :], labels[:, 1, :])
                val_crs += criterion(outputs[:, 2, :], labels[:, 2, :])
                val_crs_vel += criterion(outputs[:, 3, :], labels[:, 3, :])

            lam_vel = 1 / 1800
            lam_acc = 1 / 30
            lam_crs = 1 / 500000
            lam_crs_vel = 1 / 25000

            loss = lam_vel * val_vel + lam_acc * val_acc + lam_crs * val_crs + lam_crs_vel * val_crs_vel
            running_valid_loss += loss

        wandb.log({"valid_loss": running_valid_loss,
                   "valid_vel": val_vel,
                   "valid_acc": val_acc,
                   "valid_crs": val_crs,
                   "valid_crs_vel": val_crs_vel})

        print('#epoch:{}  train loss: {}  valid loss: {}  valid vel: {}  valid acc: {}  valid crs: {}  valid crs vel: {}'.format(epoch,
                                                                                                                    running_train_loss,
                                                                                                                    running_valid_loss,
                                                                                                                    val_vel,
                                                                                                                    val_acc,
                                                                                                                    val_crs,
                                                                                                                    val_crs_vel))

    # model_path = './BDD-X-Dataset/pre_models/model.pth'
    # torch.save(net.state_dict(), model_path)

    # 辞書オブジェクトをJSONファイルへ出力
    # with open('./out/pretrain_out.json', mode='wt', encoding='utf-8') as file:
    #     json.dump(valid_out, file, ensure_ascii=False, indent=None)


if __name__ == "__main__":
    main()
