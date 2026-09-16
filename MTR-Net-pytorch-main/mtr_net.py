import torch
import torch.nn as nn

from modules import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model,
    ReconstructionModel,
)


class MTRNet(nn.Module):
    """ 
    MTR-Net for wet/dry classification on OpenMRG.
    """

    def __init__(
            self,
            n_features,
            window_size,
            out_dim,
            kernel_size=7,
            feat_lstm_hid_dim=None,
            time_lstm_hid_dim=None,
            gru_n_layers=1,
            gru_hid_dim=150,
            forecast_n_layers=1,
            forecast_hid_dim=150,
            recon_n_layers=1,
            recon_hid_dim=150,
            dropout=0.2,
    ):
        super().__init__()

        self.conv = ConvLayer(
            n_features,
            kernel_size
        )

        self.feature_gat = FeatureAttentionLayer(
            n_features,
            window_size,
            dropout,
            embed_dim=feat_lstm_hid_dim
        )

        self.temporal_gat = TemporalAttentionLayer(
            n_features,
            window_size,
            dropout,
            embed_dim=time_lstm_hid_dim
        )

        self.gru = GRULayer(
            3 * n_features,
            gru_hid_dim,
            gru_n_layers,
            dropout
        )

        self.forecasting_model = Forecasting_Model(
            gru_hid_dim,
            forecast_hid_dim,
            out_dim,
            forecast_n_layers,
            dropout
        )

        self.recon_model = ReconstructionModel(
            window_size,
            gru_hid_dim,
            recon_hid_dim,
            out_dim,
            recon_n_layers,
            dropout
        )

        self.cls_head = nn.Linear(
            gru_hid_dim,
            1
        )

    def forward(self, x):
        x = self.conv(x)

        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)

        h_cat = torch.cat(
            [x, h_feat, h_temp],
            dim=2
        )

        _, h_end = self.gru(h_cat)
        h_end = h_end.view(
            x.shape[0],
            -1
        )

        predictions = self.forecasting_model(
            h_end
        )

        recons = self.recon_model(
            h_end
        )

        cls_logit = self.cls_head(
            h_end
        ).squeeze(-1)

        return predictions, recons, cls_logit
       