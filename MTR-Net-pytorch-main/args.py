import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="OpenMRG",
        choices=["OpenMRG"]
    )
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--normalize", type=str2bool, default=True)

    # Model
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--gru_n_layers", type=int, default=1)
    parser.add_argument("--gru_hid_dim", type=int, default=300)
    parser.add_argument("--fc_n_layers", type=int, default=3)
    parser.add_argument("--fc_hid_dim", type=int, default=300)
    parser.add_argument("--recon_n_layers", type=int, default=1)
    parser.add_argument("--recon_hid_dim", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0.3)

    # LSTM
    parser.add_argument("--feat_lstm_hid_dim", type=int, default=None)
    parser.add_argument("--time_lstm_hid_dim", type=int, default=None)

    # Test / prediction
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--scale_scores", type=str2bool, default=False)
    parser.add_argument("--use_mov_av", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--level", type=float, default=None)
    parser.add_argument("--q", type=float, default=None)
    parser.add_argument("--dynamic_pot", type=str2bool, default=False)

    return parser