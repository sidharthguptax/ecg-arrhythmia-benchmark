import torch
import torch.nn as nn


class ECG_CNN(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super(ECG_CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 180 → 90
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 90 → 45
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 45 → 22
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 22, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = ECG_CNN()
    dummy = torch.randn(32, 1, 180)  # batch of 32
    out = model(dummy)
    print("Output shape:", out.shape)  # should be (32, 5)
    print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
