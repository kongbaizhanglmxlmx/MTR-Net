import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back



class FeatureAttentionLayer(nn.Module):
    """
    输入: x.shape = (b, n, k)
    输出: h.shape = (b, n, k)
    """

    def __init__(self, n_features, window_size, dropout, embed_dim=None):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_dim = embed_dim if embed_dim is not None else 32

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        self.proj = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (b, n, k)
        b, n, k = x.shape

        # 每个 feature 单独作为一条时间序列
        x_perm = x.permute(0, 2, 1).contiguous()      # (b, k, n)
        x_reshape = x_perm.view(b * k, n, 1)          # (b*k, n, 1)

        h, _ = self.lstm(x_reshape)                   # (b*k, n, hidden_dim)
        h = self.proj(h)                              # (b*k, n, 1)
        h = self.sigmoid(h)

        h = h.view(b, k, n).permute(0, 2, 1).contiguous()   # (b, n, k)
        return h


class TemporalAttentionLayer(nn.Module):
    """
    输入: x.shape = (b, n, k)
    输出: h.shape = (b, n, k)
    """

    def __init__(self, n_features, window_size, dropout, embed_dim=None):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_dim = embed_dim if embed_dim is not None else n_features

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        self.proj = nn.Linear(self.hidden_dim, n_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (b, n, k)
        h, _ = self.lstm(x)               # (b, n, hidden_dim)
        h = self.proj(h)                  # (b, n, k)
        h = self.sigmoid(h)
        return h
# 
class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        return out, h


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)

        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out


class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)
