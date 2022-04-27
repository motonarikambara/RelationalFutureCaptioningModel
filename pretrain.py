import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
import torchvision

class MyDataset(torch.utils.data.Dataset):

  def __init__(self, label_path, transform=None):
    x = []
    y = []

    with open(label_path, 'r') as infh:
        for line in infh:
            d = line.replace('\n', '').split('\t')
            x.append(os.path.join(os.path.dirname(label_path), d[0]))
            y.append(float(d[1]))

    self.x = x
    self.y = torch.from_numpy(np.array(y)).float().view(-1, 1)

    self.transform = transform


  def __len__(self):
    return len(self.x)


  def __getitem__(self, i):
    img = PIL.Image.open(self.x[i]).convert('RGB')
    if self.transform is not None:
      img = self.transform(img)

    return img, self.y[i]


class RegressionNet(torch.nn.Module):

    def __init__(self):
        super(RegressionNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(32 * 5 * 5, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 1)


    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1, 32 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def main():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    train_data_dir = 'drive/My Drive/datasets/mnist/train_labels.tsv'
    valid_data_dir = 'drive/My Drive/datasets/mnist/valid_labels.tsv'

    trainset = MyDataset(train_data_dir, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    validset = MyDataset(valid_data_dir, transform=transform)
    validloader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=False)

    net = RegressionNet()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    train_loss = []
    valid_loss = []

    for epoch in range(50):
        # 学習
        net.train()
        running_train_loss = 0.0
        with torch.set_grad_enabled(True):
            for data in trainloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_train_loss += loss.item()
                loss.backward()
                optimizer.step()

        train_loss.append(running_train_loss / len(trainset))

        # 検証
        net.eval()
        running_valid_loss = 0.0
        with torch.set_grad_enabled(False):
            for data in validloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                running_valid_loss += loss.item()

        valid_loss.append(running_valid_loss / len(validset))

        print('#epoch:{}\ttrain loss: {}\tvalid loss: {}'.format(epoch,
                                                        running_train_loss / len(train_loss),
                                                        running_valid_loss / len(valid_loss)))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(train_loss, label='train')
    ax.plot(valid_loss, label='valid')
    fig.show()


if __name__ == "__main__":
    main()
