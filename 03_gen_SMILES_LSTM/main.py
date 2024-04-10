import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from smiles.lstm.smiles_lit_model import SmilesLitModel  # Make sure to import the SmilesLitModel class
from smiles.lstm.smiles_data_module import SmilesDataModule  # Make sure to import the SmilesDataModule class

def main():
    # Define hyperparameters
    vocab_size = 1000  # Example size, replace with your actual vocab size
    batch_size = 256
    learning_rate = 0.001
    max_epochs = 10
    data_dir = './data/'  # Update this path to where your data is stored

    # Initialize the model
    model = SmilesLitModel(vocab_size=vocab_size, learning_rate=learning_rate, batch_size=batch_size)

    # Initialize the data module
    data_module = SmilesDataModule(data_dir=data_dir, batch_size=batch_size)

    # Set up logging and checkpoints
    logger = TensorBoardLogger("tb_logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints", monitor="val_loss")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=max_epochs, logger=logger, callbacks=[checkpoint_callback, lr_monitor])

    # Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()