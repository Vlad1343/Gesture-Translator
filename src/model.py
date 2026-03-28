# src/model.py
import torch
import torch.nn as nn

class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=5, dropout=0.3):
        super(GestureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # classifier
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_length, input_size]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # out: [batch, seq_len, hidden]
        out = out[:, -1, :]               # take last time step
        out = self.fc(out)                # [batch, num_classes]
        return out