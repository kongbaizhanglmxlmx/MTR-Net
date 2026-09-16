import os
import json

import torch
import torch.nn as nn

from args import get_parser
from utils import *
from mtr_net import MTRNet
from prediction import Predictor
from training import Trainer



BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_ID = "25062026_213707"

MODEL_DIR = os.path.join(
    BASE_DIR,
    "output",
    "OpenMRG",
    MODEL_ID
)

if __name__ == "__main__":


    parser = get_parser()
    args = parser.parse_args()

    dataset = args.dataset
    window_size = args.lookback
    normalize = args.normalize

    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda

    # Trainer 仅用于加载和测试，不执行 fit()
    n_epochs = 1
    print_every = 1

    args_summary = str(args.__dict__)

    print(args_summary)

    save_path = MODEL_DIR

    model_path = os.path.join(
        save_path,
        "model.pt"
    )


    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"没有找到模型文件：\n{model_path}"
        )

    print("=" * 70)
    print("Load saved model")
    print("=" * 70)

    print(f"Model folder: {save_path}")
    print(f"Model file:   {model_path}")

    if dataset != "OpenMRG":
        raise ValueError(
            "Only OpenMRG is supported."
        )

    output_path = os.path.join(
        "output",
        "OpenMRG"
    )

    (x_train, y_train), (
        x_test,
        y_test
    ) = get_data(
        dataset,
        normalize=normalize
    )

    if y_test is None:
        raise ValueError(
            "没有读取到 y_test，"
            "Model 无法计算 classification loss。"
        )

  
    x_train = torch.from_numpy(
        x_train
    ).float()

    x_test = torch.from_numpy(
        x_test
    ).float()

    n_features = x_train.shape[1]

    
    target_dims = get_target_dims(
        dataset
    )

    if target_dims is None:

        out_dim = n_features

        print(
            f"Will forecast and reconstruct all "
            f"{n_features} input features"
        )

    elif type(target_dims) == int:

        out_dim = 1

        print(
            f"Will forecast and reconstruct "
            f"input feature: {target_dims}"
        )

    else:

        out_dim = len(
            target_dims
        )

        print(
            f"Will forecast and reconstruct "
            f"input features: {target_dims}"
        )

  
    train_dataset = SlidingWindowDataset(
        x_train,
        window_size,
        target_dims,
        labels=y_train,
        label_mode="last"
    )

    test_dataset = SlidingWindowDataset(
        x_test,
        window_size,
        target_dims,
        labels=y_test,
        label_mode="last"
    )

  
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        batch_size,
        val_split,
        shuffle_dataset,
        test_dataset=test_dataset
    )

   
    model = MTRNet(
        n_features=n_features,
        window_size=window_size,
        out_dim=out_dim,
        kernel_size=args.kernel_size,
        feat_lstm_hid_dim=args.feat_lstm_hid_dim,
        time_lstm_hid_dim=args.time_lstm_hid_dim,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
    )

   
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=init_lr
    )

    
    forecast_criterion = nn.MSELoss()

    recon_criterion = nn.MSELoss()

    
    cls_criterion = nn.BCEWithLogitsLoss()

  
    lambda_recon = 1.0
    lambda_cls = 0.5

    
    log_dir = os.path.join(
        output_path,
        "logs"
    )

    os.makedirs(
        log_dir,
        exist_ok=True
    )

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,

        forecast_criterion,
        recon_criterion,
        cls_criterion,

        lambda_recon,
        lambda_cls,

        use_cuda,

        save_path,
        log_dir,

        print_every,

        False,

        args_summary,

        early_stop_patience=7
    )

   
    print()
    print("=" * 70)
    print("Loading model.pt ...")
    print("=" * 70)

    trainer.load(
        model_path
    )

    print("model.pt loaded successfully.")

   
    print()
    print("=" * 70)
    print("Testing Model ...")
    print("=" * 70)

    test_loss = trainer.evaluate(
        test_loader
    )

    
    print(
        f"Test forecast loss: "
        f"{test_loss[0]:.5f}"
    )

    print(
        f"Test reconstruction loss: "
        f"{test_loss[1]:.5f}"
    )

    print(
        f"Test classification loss: "
        f"{test_loss[2]:.5f}"
    )

    print(
        f"Test total loss: "
        f"{test_loss[3]:.5f}"
    )

   
    level = (
        args.level
        if args.level is not None
        else 0.90
    )

    q = (
        args.q
        if args.q is not None
        else 0.001
    )

    reg_level = 0

 
    

    

  
    prediction_args = {

        "dataset": dataset,

        "target_dims": target_dims,

        "scale_scores": args.scale_scores,

        "level": level,

        "q": q,

        "dynamic_pot": args.dynamic_pot,

        "use_mov_av": args.use_mov_av,

        "gamma": args.gamma,

        "reg_level": reg_level,

       
        "save_path": save_path
    }

    predictor = Predictor(
        trainer.model,
        window_size,
        n_features,
        prediction_args
    )

   
    label = y_test[
        window_size:
    ]

 
    print()
    print("=" * 70)
    print("Running final anomaly detection ...")
    print("=" * 70)

    predictor.predict_anomalies(
        x_train,
        x_test,
        label
    )

    
    config_path = os.path.join(
        save_path,
        "config.txt"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            args.__dict__,
            f,
            indent=2,
            ensure_ascii=False
        )

    
    print()
    print("=" * 70)
    print("All testing finished.")
    print("=" * 70)

    print(save_path)