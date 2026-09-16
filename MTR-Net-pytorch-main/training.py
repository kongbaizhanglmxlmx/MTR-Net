import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """Training and evaluation utilities for MTR-Net.

    :param model: model
    :param optimizer: Optimizer used to minimize the loss function
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param target_dims: dimension of input features to forecast and reconstruct
    :param n_epochs: Number of iterations/epochs
    :param batch_size: Number of windows in a single batch
    :param init_lr: Initial learning rate of the module
    :param forecast_criterion: Loss to be used for forecasting
    :param recon_criterion: Loss to be used for reconstruction
    :param cls_criterion: Loss to be used for wet/dry classification
    :param lambda_recon: Weight for reconstruction loss
    :param lambda_cls: Weight for classification loss
    :param boolean use_cuda: To be run on GPU or not
    :param dload: Download directory where models are to be dumped
    :param log_dir: Directory where SummaryWriter logs are written to
    :param print_every: At what epoch interval to print losses
    :param log_tensorboard: Whether to log loss++ to tensorboard
    :param args_summary: Summary of args that will also be written to tensorboard if log_tensorboard
    """

    def __init__(
        self,
        model,
        optimizer,
        window_size,
        n_features,
        target_dims=None,
        n_epochs=200,
        batch_size=256,
        init_lr=0.001,
        forecast_criterion=nn.MSELoss(),
        recon_criterion=nn.MSELoss(),
        cls_criterion=nn.BCEWithLogitsLoss(),
        lambda_recon=1.0,
        lambda_cls=0.5,
        use_cuda=True,
        dload="",
        log_dir="output/",
        print_every=1,
        log_tensorboard=True,
        args_summary="",
       
        early_stop_patience=7,
         
    ):

        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr

        self.forecast_criterion = forecast_criterion
        self.recon_criterion = recon_criterion
        self.cls_criterion = cls_criterion

        self.lambda_recon = lambda_recon
        self.lambda_cls = lambda_cls

        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.dload = dload
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard
        self.early_stop_patience = early_stop_patience

        self.no_improve_count = 0
        self.best_epoch = -1
        self.best_state_dict = None
    

        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "train_recon": [],
            "train_cls": [],
            "val_total": [],
            "val_forecast": [],
            "val_recon": [],
            "val_cls": [],
        }
        self.epoch_times = []
        self.best_val_loss = np.inf

        if self.device == "cuda":
            self.model.cuda()

        if self.log_tensorboard:
            self.writer = SummaryWriter(f"{log_dir}")
            self.writer.add_text("args_summary", args_summary)
       
    def fit(self, train_loader, val_loader=None):
        """Train model for self.n_epochs.
        Train and validation (if validation loader given) losses stored in self.losses
        """

        init_train_loss = self.evaluate(train_loader)
        print(f"Init total train loss: {init_train_loss[3]:.5f}")
        
        if val_loader is not None:
            init_val_loss = self.evaluate(val_loader)
            print(f"Init total val loss: {init_val_loss[3]:.5f}")
       
        print(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()

        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()

            forecast_b_losses = []
            recon_b_losses = []
            cls_b_losses = []

            for x, y_forecast, y_cls in train_loader:
                x = x.to(self.device)
                y_forecast = y_forecast.to(self.device)
                y_cls = y_cls.to(self.device).float()
           
                self.optimizer.zero_grad()

                preds, recons, cls_logit = self.model(x)
               
                x_recon_target = x
                if self.target_dims is not None:
                    x_recon_target = x[:, :, self.target_dims]
                    y_forecast = y_forecast[:, :, self.target_dims]

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y_forecast.ndim == 3:
                    y_forecast = y_forecast.squeeze(1)

                forecast_loss = torch.sqrt(self.forecast_criterion(y_forecast, preds))
                recon_loss = torch.sqrt(self.recon_criterion(x_recon_target, recons))
                cls_loss = self.cls_criterion(cls_logit, y_cls)


                loss = forecast_loss + self.lambda_recon * recon_loss + self.lambda_cls * cls_loss
               
                loss.backward()
                self.optimizer.step()

                forecast_b_losses.append(forecast_loss.item())
                recon_b_losses.append(recon_loss.item())
                cls_b_losses.append(cls_loss.item())

            forecast_b_losses = np.array(forecast_b_losses)
            recon_b_losses = np.array(recon_b_losses)
            cls_b_losses = np.array(cls_b_losses)

            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())
            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())

            cls_epoch_loss = cls_b_losses.mean()

            total_epoch_loss = (
                forecast_epoch_loss
                + self.lambda_recon * recon_epoch_loss
                + self.lambda_cls * cls_epoch_loss
            )

            self.losses["train_forecast"].append(forecast_epoch_loss)
            self.losses["train_recon"].append(recon_epoch_loss)
            self.losses["train_cls"].append(cls_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)

            
            forecast_val_loss, recon_val_loss, total_val_loss = "NA", "NA", "NA"
            
            if val_loader is not None:
                forecast_val_loss, recon_val_loss, cls_val_loss, total_val_loss = self.evaluate(val_loader)

                self.losses["val_forecast"].append(forecast_val_loss)
                self.losses["val_recon"].append(recon_val_loss)
                self.losses["val_cls"].append(cls_val_loss)
                self.losses["val_total"].append(total_val_loss)

                if total_val_loss < self.best_val_loss:
                    self.best_val_loss = total_val_loss
                    self.best_epoch = epoch + 1
                    self.best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    self.no_improve_count = 0
                    self.save("model.pt")
                else:
                    self.no_improve_count += 1
            
                if self.no_improve_count >= self.early_stop_patience:
                    print(
                        f"[Early Stop] epoch={epoch + 1}, "
                        f"best_epoch={self.best_epoch}, "
                        f"best_val_loss={self.best_val_loss:.5f}"
                    )
                    break
                    #


            if self.log_tensorboard:
                self.write_loss(epoch)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            if epoch % self.print_every == 0:
                
                current_lr = self.optimizer.param_groups[0]["lr"]
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"recon_loss = {recon_epoch_loss:.5f}, "
                    f"cls_loss = {cls_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )

                if val_loader is not None:
                    s += (
                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "
                        f"val_recon_loss = {recon_val_loss:.5f}, "
                        f"val_cls_loss = {cls_val_loss:.5f}, "
                        f"val_total_loss = {total_val_loss:.5f}"
                    )

                s += f" [{epoch_time:.1f}s]"
                print(s)


        if val_loader is None:
            self.save("model.pt")
            
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
           

        train_time = int(time.time() - train_start)
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
        print(f"-- Training done in {train_time}s.")

    def evaluate(self, data_loader):
        """Evaluate model

        :param data_loader: data loader of input data
        :return: forecasting loss, reconstruction loss, classification loss, total loss
        """
        self.model.eval()

        forecast_losses = []
        recon_losses = []
        cls_losses = []

        with torch.no_grad():
            for x, y_forecast, y_cls in data_loader:
                x = x.to(self.device)
                y_forecast = y_forecast.to(self.device)
                y_cls = y_cls.to(self.device).float()

                preds, recons, cls_logit = self.model(x)

                x_recon_target = x
                if self.target_dims is not None:
                    x_recon_target = x[:, :, self.target_dims]
                    y_forecast = y_forecast[:, :, self.target_dims]

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y_forecast.ndim == 3:
                    y_forecast = y_forecast.squeeze(1)
       
                forecast_loss = torch.sqrt(self.forecast_criterion(y_forecast, preds))
                recon_loss = torch.sqrt(self.recon_criterion(x_recon_target, recons))
                cls_loss = self.cls_criterion(cls_logit, y_cls)

                forecast_losses.append(forecast_loss.item())
                recon_losses.append(recon_loss.item())
                cls_losses.append(cls_loss.item())

        forecast_losses = np.array(forecast_losses)
        recon_losses = np.array(recon_losses)
        cls_losses = np.array(cls_losses)

        forecast_loss = np.sqrt((forecast_losses ** 2).mean())
        recon_loss = np.sqrt((recon_losses ** 2).mean())
        cls_loss = cls_losses.mean()

        total_loss = (
            forecast_loss
            + self.lambda_recon * recon_loss
            + self.lambda_cls * cls_loss
        )

        return forecast_loss, recon_loss, cls_loss, total_loss
      
    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as, `dload` serves as the download directory
        """
        path = self.dload + "/" + file_name
        if not os.path.exists(self.dload):
            os.makedirs(self.dload, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Loads the model's parameters from the path mentioned
        :param path: Should contain pickle file
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)