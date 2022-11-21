# model.py
import torch
import torch.optim as optim
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from config import *
from utils import *
from sklearn.metrics import classification_report

class SegmentationModule(pl.LightningModule):

    def __init__(self, model, loss_type, encoder, encoder_weights, step_lr):
        super().__init__()
        self.save_hyperparameters()
        if loss_type == 'focal':
            self.loss = smp.losses.FocalLoss(LOSS_MODE)
        elif loss_type == 'dice':
            self.loss = smp.losses.DiceLoss(LOSS_MODE)
        elif loss_type == 'ce':
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.3, 0.3, 0.2]))
        else:
            raise NotImplementedError
        if model == 'unet':
            self.model = smp.Unet(
                            encoder_name=encoder,        
                            encoder_weights=encoder_weights,    
                            in_channels=1,                  
                            classes=4,                      
                        )
        elif model == 'fpn':
            self.model = smp.FPN(
                            encoder_name=encoder,       
                            encoder_weights=encoder_weights,  
                            in_channels=1,                 
                            classes=4,                     
                        )
        elif model == 'linknet':
            self.model = smp.Linknet(
                            encoder_name=encoder,       
                            encoder_weights=encoder_weights,   
                            in_channels=1,                
                            classes=4,                    
                        )

    def predict(self, x):
        with torch.no_grad():
            pred = self.forward(x)
        return pred.argmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        return torch.softmax(x, dim=1)
    
    def training_step(self, batch, batch_idx): 
        x, y = batch # N * C * H * W, N * H * W
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return {'train_loss': loss, 'y_hat': y_hat.argmax(dim=1), 'y': y.clone().detach()}
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return {'val_loss': loss, 'y_hat': y_hat.argmax(dim=1), 'y': y.clone().detach()}
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return {'test_loss': loss, 'y_hat': y_hat.argmax(dim=1), 'y': y.clone().detach()}
    
    def validation_epoch_end(self, val_batch_outputs):
        loss = torch.stack([x['val_loss'] for x in val_batch_outputs]).mean()
        y_hat = torch.cat([x['y_hat'] for x in val_batch_outputs], dim=0).flatten()
        y = torch.cat([x['y'] for x in val_batch_outputs], dim=0).flatten()
        print(classification_report(y.cpu().numpy(), y_hat.cpu().numpy()))


    def test_epoch_end(self, test_batch_outputs):
        loss = torch.stack([x['test_loss'] for x in test_batch_outputs]).mean()
        y_hat = torch.cat([x['y_hat'] for x in test_batch_outputs], dim=0).flatten()
        y = torch.cat([x['y'] for x in test_batch_outputs], dim=0).flatten()
        print(classification_report(y.cpu().numpy(), y_hat.cpu().numpy()))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-2)
        return optimizer
