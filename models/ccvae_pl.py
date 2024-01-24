import lightning as L
from .ccvae import CCVAE
import torch
from torch.utils.data import DataLoader

class CCVAE_L(L.LightningModule):
    
    def __init__(self, in_shape,
                z_dim,
                num_classes,
                lr,
                prior_fn,
                train_dataset=None, 
                valid_dataset=None,
                test_dataset = None
                ):
        super().__init__()
        self.lr = lr
        self.model = CCVAE(in_shape=in_shape,
                           z_dim=z_dim,
                           num_classes=num_classes,
                           prior_fn=prior_fn)
        
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        

    def training_step(self, batch, batch_idx):
        x, y = self.in_transform(batch)
        
        neg_elbo, log_qyzc, log_qyzc_, log_qyx, log_pxz, kl, log_py = self.model.sup(x,y)
        self.log("loss/train/neg_elbo", neg_elbo)
        self.log("loss/train/log_qyzc", log_qyzc)
        self.log("loss/train/log_qyzc_", log_qyzc_)
        self.log("loss/train/log_qyx", log_qyx)
        self.log("loss/train/log_pxz", log_pxz)
        self.log("loss/train/log_py", log_py)
        self.log("loss/train/kl", kl)

        return neg_elbo
    
    def validation_step(self, batch, batch_idx):
        x, y = self.in_transform(batch)
        neg_elbo, log_qyzc, log_qyzc_, log_qyx, log_pxz, kl, log_py = self.model.sup(x,y)
        
        self.log("loss/val/neg_elbo", neg_elbo)
        self.log("loss/val/log_qyzc", log_qyzc)
        self.log("loss/val/log_qyzc_", log_qyzc_)
        self.log("loss/val/log_qyx", log_qyx)
        self.log("loss/val/log_pxz", log_pxz)
        self.log("loss/val/log_py", log_py)
        self.log("loss/val/kl", kl)

        return neg_elbo

    def prediction(self, dataset=None):
        
        if dataset is None: dataset = self.test_dataset
        dataloader = self._dataloader(dataset)

        x_list = []
        y_list = []
        r_list = []
        loc_list = []
        scale_list = []

        for batch in dataloader:
            x, y = self.in_transform(batch)
            x_list.append(x)
            y_list.append(y)
            r_list.append(self.model.reconstruct_img(x))
            loc, scale = self.model.get_latent(x)
            
            loc_list.append(loc)
            scale_list.append(scale)

        return torch.cat(x_list, dim=0), torch.cat(y_list, dim=0), torch.cat(r_list, dim=0), torch.cat(loc_list, dim=0), torch.cat(scale_list, dim=0)

    def predict_step(self, batch):
        x, y = self.in_transform(batch)
        r = self.model.reconstruct_img(x)
        loc, scale = self.model.get_latent(x)
        return x, r, loc, scale

    def in_transform(self, batch):
        x , y = batch
        x = torch.cat(x, dim=1)
        x = torch.permute(x, (0,2,1))

        return x, y
    
    def configure_optimizers(self):
        # Cosine Annealing LR Scheduler

        optimizer = torch.optim.Adam(
            list(
                filter(
                    lambda p: p.requires_grad,
                    self.model.parameters(),
                )
            ),
            lr=self.lr,
        )
        # scheduler = self._get_scheduler(optimizer=optimizer)
        
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            # "monitor": "val_loss",
            # "interval":"epoch"
        }
    
    def train_dataloader(self):
        if self.train_dataset is not None:
            return self._dataloader(self.train_dataset, shuffle=True)
        else:
            return None
        
    def val_dataloader(self):
        if self.valid_dataset is not None:
            return self._dataloader(self.valid_dataset, shuffle=False)
        else:
            return None
        
    def predict_dataloader(self):
        if self.test_dataset is not None:
            return self._dataloader(self.test_dataset, shuffle=False)
        else:
            return None

    def _dataloader(self, dataset, batch_size=128, shuffle=False ):

        return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            )
    
    def accuracy(self, dataset=None):
        if dataset is None: dataset = self.test_dataset
        dataloader = self._dataloader(dataset)

        acc = 0
        for batch in dataloader:
            x, y = self.in_transform(batch)
            batch_acc = self.model.classifier_acc(x, y)
            acc+=batch_acc

        return acc/len(dataloader)
    
    def label_prediction(self, dataset = None, prob=False):
        
        if dataset is None: dataset = self.test_dataset
        dataloader = self._dataloader(dataset)

        y_pred = []
        y_true = []

        for batch in dataloader:
            x, y = self.in_transform(batch)
            y_pred.append(self.model.classifier_pred(x, prob))
            y_true.append(y)


        return torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0)

    def latent_walk(self, x):
        return self.model.latent_walk(x)



