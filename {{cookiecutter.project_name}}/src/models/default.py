import lightning as L
import torchmetrics
from torch import nn

class DefaultNN(L.LightningModule):
    def __init__(self, criterion: nn.Module, in_channels: int = 1, out_channels: int = 10):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = criterion
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_channels),
        )

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc.update(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.test_acc.update(logits, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc)
        return loss

