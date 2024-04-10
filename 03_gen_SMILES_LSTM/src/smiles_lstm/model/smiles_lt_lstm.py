import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from smiles_lstm.model.smiles_lstm import RNN, SmilesLSTM
from smiles_lstm.model.smiles_dataset import Dataset
from smiles_lstm.model.smiles_vocabulary import SMILESTokenizer, Vocabulary
import smiles_lstm.utils.load as load

class SmilesLitModel(pl.LightningModule):
    def __init__(self, vocab_size, learning_rate=0.0001, batch_size=250, lr_scheduler_type="StepLR", gamma=0.8):
        super().__init__()
        self.save_hyperparameters()
        # Assuming vocab_size is obtained from the Vocabulary instance.
        network_params = {
            "voc_size": vocab_size,
            "layer_size": 512,
            "num_layers": 3,
            "cell_type": 'lstm',
            "embedding_layer_size": 256,
            "dropout": 0.0,
            "layer_normalization": False
        }
        self.model = SmilesLSTM(vocabulary=Vocabulary(), tokenizer=SMILESTokenizer(), network_params=network_params)
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.gamma = gamma
        self.batch_size = batch_size
        self.loss_fn = torch.nn.NLLLoss(reduction="none")

    def forward(self, input_vector, hidden_state=None):
        return self.model.network(input_vector, hidden_state)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs, _ = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs, _ = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.network.parameters(), lr=self.learning_rate)
        if self.lr_scheduler_type == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
        elif self.lr_scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        else:
            raise ValueError("Unsupported learning rate scheduler type")
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
