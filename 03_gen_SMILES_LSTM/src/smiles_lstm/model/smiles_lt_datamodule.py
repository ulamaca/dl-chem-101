import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from smiles_lstm.model.smiles_lstm import RNN, SmilesLSTM
from smiles_lstm.model.smiles_dataset import Dataset
from smiles_lstm.model.smiles_vocabulary import SMILESTokenizer, Vocabulary
import smiles_lstm.utils.load as load

class SmilesDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=250):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Assuming `load.smiles` function loads SMILES and their targets into (smiles, targets) tuples.
        train_smiles, test_smiles = load.smiles(self.data_dir)
        self.train_dataset = Dataset(train_smiles)
        self.test_dataset = Dataset(test_smiles)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
