import torch
import torch.nn as nn
import torch.nn.functional as F

class BigCNN(nn.Module):
    model_name = "BigCNN"

    def __init__(self) -> None:
        super().__init__()
        # 8*8 board: big kernels -> only few channels
        # start with 13 channels
        # 6 white pieces, 6 black pieces, and no piece = 13
        self.conv1_8_8 = nn.Conv2d(
            in_channels=13, out_channels=13, kernel_size=1, padding="same"
        )
        self.dropout1_8_8 = nn.Dropout2d(0.5)
        self.conv2_8_8 = nn.Conv2d(
            in_channels=13, out_channels=16, kernel_size=3, padding="same"
        )
        self.dropout2_8_8 = nn.Dropout2d(0.5)
        self.conv3_8_8 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, padding="same"
        )

        # If we want same padding -> https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121
        # Doesn't seem a problem here as the size fits at the moment.
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # 4*4 boards
        self.conv1_4_4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=2, padding="same"
        )
        self.dropout1_4_4 = nn.Dropout2d(0.5)
        self.conv2_4_4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=2, padding="same"
        )
        self.dropout2_4_4 = nn.Dropout2d(0.5)
        self.conv3_4_4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=2, padding="same"
        )

        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # 2*2 boards
        # this seems serious overkill, I need to plot the response of these things check if
        # they end up producing useful patterns.
        self.conv1_2_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=1, padding="same"
        )
        self.dropout1_2_2 = nn.Dropout2d(0.5)
        self.conv2_2_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=1, padding="same"
        )
        self.dropout2_2_2 = nn.Dropout2d(0.5)
        self.conv3_2_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=1, padding="same"
        )

        # 2*2 image * 128 channels = 4*128=512
        self.output = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        x = F.relu(self.conv1_8_8(x))
        x = self.dropout1_8_8(x)
        x = F.relu(self.conv2_8_8(x))
        x = self.dropout2_8_8(x)
        x = F.relu(self.conv3_8_8(x))

        x = self.max_pool1(x)

        x = F.relu(self.conv1_4_4(x))
        x = self.dropout1_4_4(x)
        x = F.relu(self.conv2_4_4(x))
        x = self.dropout2_4_4(x)
        x = F.relu(self.conv3_4_4(x))

        x = self.max_pool2(x)

        x = F.relu(self.conv1_2_2(x))
        x = self.dropout1_2_2(x)
        x = F.relu(self.conv2_2_2(x))
        x = self.dropout2_2_2(x)
        x = F.relu(self.conv3_2_2(x))

        x = torch.flatten(x, start_dim=1)
        x = self.output(x)

        return x
