import pytorch_lightning as pl

from torch.utils.data import random_split, DataLoader

from .lmx_dataset import LMXDataset
from .lmx_tokenizer import LMXTokenizer


class LMXDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for handling LMX sequence data processing.

    This module handles dataset preparation, splitting, and DataLoader creation
    for training, validation, and testing of LMX sequence models.
    """

    def __init__(
        self, lmx_data: list[str], tokenizer: LMXTokenizer, batch_size, max_length
    ) -> None:
        """Initialize the LMXDataModule with data and processing parameters.

        Args:
            lmx_data (list[str]): List of raw LMX sequences
            tokenizer (LMXTokenizer): Custom tokenizer for LMX sequence processing
            batch_size (int): Number of samples per batch
            max_length (int): Maximum sequence length
        """
        super().__init__()
        self.lmx_data = lmx_data  # List of LMX sequences
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None) -> None:
        """Prepare and split dataset into train/validation/test subsets.

        Args:
            stage (str, optional): Used by PyTorch Lightning for stage-specific setup.
            Defaults to None.
        """
        dataset = LMXDataset(self.lmx_data, self.tokenizer, max_length=self.max_length)

        total_size = len(dataset)
        train_size = int(0.9 * total_size)
        val_size = int(0.05 * total_size)
        test_size = total_size - train_size - val_size  # Ensures all data is used

        self.train_data, self.val_data, self.test_data = random_split(
            dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader with shuffling.

        Returns:
            DataLoader: Configured DataLoader for training data
        """
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader without shuffling.

        Returns:
            DataLoader: Configured DataLoader for validation data
        """
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader without shuffling.

        Returns:
            DataLoader: Configured DataLoader for test data
        """
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )
