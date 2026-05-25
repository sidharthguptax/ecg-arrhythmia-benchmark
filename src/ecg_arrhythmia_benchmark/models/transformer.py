import torch
import torch.nn as nn

from ecg_arrhythmia_benchmark.models.cnn import ECG_CNN


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.3, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ECG_Transformer(ECG_CNN):
    def __init__(
        self,
        num_classes=5,
        dropout=0.3,
        d_model=64,
        nhead=4,
        num_layers=2,
        max_len=512,
    ):
        super().__init__(num_classes=num_classes, dropout=dropout)

        self.feature_projection = nn.LazyLinear(d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=max_len
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.transformer_classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.permute(0, 2, 1)
        x = self.feature_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        x = x.mean(dim=1)
        x = self.transformer_classifier(x)
        return x


if __name__ == "__main__":
    model = ECG_Transformer()
    dummy = torch.randn(32, 1, 180)
    out = model(dummy)
    print("Output shape:", out.shape)
    print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
