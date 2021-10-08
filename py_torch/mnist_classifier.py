import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


class SmallAndSmartModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 28, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(28, 10, kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.dropout1 = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(250, 18)
        self.dropout2 = torch.nn.Dropout(0.08)
        self.fc2 = torch.nn.Linear(18, 10)

    def prepare_data(self) -> None:
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)

    def train_dataloader(self) -> DataLoader:
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
        self.train_set, self.val_set = random_split(mnist_train, [55000, 5000])
        return DataLoader(self.train_set, batch_size=128)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=128)

    def test_dataloader(self) -> DataLoader:
        mnist_test = MNIST(os.getcwd(), train=False, download=False, transform=transforms.ToTensor())
        return DataLoader(mnist_test, batch_size=128)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.leaky_relu(self.dropout2(x))
        return F.softmax(self.fc2(x))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, labels = batch
        pred = self.forward(x)
        loss = F.nnl_loss(pred, labels)
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}


my_trainer = pl.Trainer(gpus=1, max_nb_epochs=100)
model = SmallAndSmartModel()
my_trainer.fit(model)
