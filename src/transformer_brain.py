import torch
import torch.nn as nn

class TaskTransformer(nn.Module):

    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # completed / not completed
        )

    def forward(self, x):

        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)

        return self.head(x)