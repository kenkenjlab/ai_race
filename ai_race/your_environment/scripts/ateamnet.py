import torch
import torch.nn as nn
import torch.nn.functional as F

class ATeamNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ATeamNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)  # RGB 3ch
        self.conv2 = nn.Conv2d(16, 16, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, padding=1)
        self.fc1 = nn.Linear(int(input_size * 16 / 64), 64)       # Here input tensor size is 1/64 of original image, but has 16ch.

        # Dueling network
        self.fc2_adv = nn.Linear(64, output_size)
        self.fc2_val = nn.Linear(64, 1)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2_adv.weight)
        nn.init.kaiming_normal_(self.fc2_val.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)

        # Dueling network
        adv = self.fc2_adv(x)
        val = self.fc2_val(x).expand(-1, adv.size(1))
        x = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return x

if __name__ == '__main__':
    img_size = (80, 32)  # W, H
    net = ATeamNet(img_size[0] * img_size[1], 3)
    print(net)