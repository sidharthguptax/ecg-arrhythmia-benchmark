import torch
import torch.nn as nn


class ECG_BiLSTM(nn.Module):
    def __init__(self, num_classes=5, hidden_size=128, num_layers=2, dropout=0.3):
        super(ECG_BiLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 because bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, 1, 180) → (batch, 180, 1)
        x = x.permute(0, 2, 1)

        # lstm_out: (batch, 180, hidden_size*2)
        lstm_out, _ = self.lstm(x)

        # take last time step
        x = lstm_out[:, -1, :]  # (batch, hidden_size*2)

        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = ECG_BiLSTM()
    dummy = torch.randn(32, 1, 180)
    out = model(dummy)
    print("Output shape:", out.shape)  # should be (32, 5)
    print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
