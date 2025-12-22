import lightning as L
import torch, medmnist
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from medmnist import INFO

class MedMNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        name: str = "pathmnist",
        size: int = 28,
        as_rgb: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.name = name.lower()
        self.size = size
        self.as_rgb = as_rgb
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Dataset info
        self.info = INFO[self.name]
        self.DataClass = getattr(medmnist, self.info["python_class"])

        # Transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def prepare_data(self):
        self.DataClass(
            split="train",
            download=True,
            root=self.data_dir,
            size=self.size,
        )
        self.DataClass(
            split="test",
            download=True,
            root=self.data_dir,
            size=self.size,
        )

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            full_train = self.DataClass(
                split="train",
                root=self.data_dir,
                transform=self.transform,
                size=self.size,
            )

            val_size = int(0.1 * len(full_train))
            train_size = len(full_train) - val_size

            self.train_set, self.val_set = random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test" or stage is None:
            self.test_set = self.DataClass(
                split="test",
                root=self.data_dir,
                transform=self.transform,
                size=self.size,
            )

        if stage == "predict":
            self.predict_set = self.DataClass(
                split="test",
                root=self.data_dir,
                transform=self.transform,
                size=self.size,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
