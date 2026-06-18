import torch
import torch.nn as nn

LSTM_FEATURES = [
    "close_pct_1",
    "close_pct_3",
    "rsi_14",        
    "macd_hist",     
    "bb_position",   
    "vol_ratio",     
    "atr_14",        
    "imoex",         
    "usd_rub", 
]

LOOKBACK = 60

class StockLSTM(nn.Module):
    def __init__(
        self,
        input_size:  int   = len(LSTM_FEATURES),
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        attn_weights = self.attention(out)
        attn_output = torch.sum(out * attn_weights, dim=1)
        return self.head(attn_output).squeeze(1)