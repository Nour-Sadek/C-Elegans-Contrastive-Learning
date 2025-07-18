import torch
import pickle

import optuna
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader

from motif_based_encoder import MotifBasedEncoder
from using_the_model import (infoNCE_loss, get_encoder_inputs_from_batch, get_seqs_outside_batch,
                             evaluate_representations, split_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load back the valid genes
file_path = f"./valid_genes_for_encoder_fam_size_8.pkl"
with open(file_path, "rb") as f:
    valid_genes = pickle.load(f)
print(f"Hyperparameter tuning of the model will start.")

# Split the data into training and validation data sets
train_set, val_set = split_data(valid_genes)


# Create a simple Dataset class to wrap the list of genes
class PreBatchedDataset(Dataset):
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


class LitMotifBasedEncoder(pl.LightningModule):
    def __init__(self, train_set, val_set, batch_size, learning_rate, temperature, family_size, target_set_size):
        super().__init__()
        self.save_hyperparameters(ignore=["train_set", "val_set", "family_size"])

        # input data sets
        self.train_set = train_set
        self.val_set = val_set

        # MotifBasedEncoder
        self.encoder = MotifBasedEncoder(num_PWMs=256, PWM_width=15, window=10, num_bases=4)
        self.family_size = family_size

    def forward(self, x):
        seqs_embeddings = self.encoder(x)
        return seqs_embeddings

    def training_step(self, batch, batch_idx):
        num_genes_in_batch = len(batch)
        inputs = get_encoder_inputs_from_batch(self.train_set, batch, self.family_size)
        # Fetch the extra negative sequences from genes outside the current batch
        num_non_batch_negative_seqs_to_add = self.hparams.target_set_size - num_genes_in_batch
        negative_seqs_outside_batch = get_seqs_outside_batch(self.train_set, batch,
                                                             num_non_batch_negative_seqs_to_add)
        # Add those sequences to the end of the <inputs>
        inputs = torch.cat([inputs, torch.stack(negative_seqs_outside_batch)], dim=0)
        inputs = inputs.to(device)
        seqs_embeddings = self.encoder(inputs)
        loss = infoNCE_loss(seqs_embeddings, self.family_size, num_genes_in_batch, self.hparams.target_set_size,
                            self.hparams.temperature)
        self.log("train_loss", loss, batch_size=num_genes_in_batch)
        return loss

    def validation_step(self, batch, batch_idx):
        num_genes_in_batch = len(batch)
        inputs = get_encoder_inputs_from_batch(self.val_set, batch, self.family_size)
        # Fetch the extra negative sequences from genes outside the current batch
        num_non_batch_negative_seqs_to_add = self.hparams.target_set_size - num_genes_in_batch
        negative_seqs_outside_batch = get_seqs_outside_batch(self.val_set, batch,
                                                             num_non_batch_negative_seqs_to_add)
        # Add those sequences to the end of the <inputs>
        inputs = torch.cat([inputs, torch.stack(negative_seqs_outside_batch)], dim=0)
        inputs = inputs.to(device)
        seqs_embeddings = self.encoder(inputs)
        loss = infoNCE_loss(seqs_embeddings, self.family_size, num_genes_in_batch, self.hparams.target_set_size,
                            self.hparams.temperature)
        accuracy = evaluate_representations(self.encoder, self.val_set, self.family_size, self.hparams.batch_size,
                                            self.hparams.target_set_size, self.hparams.temperature)
        self.log("val_loss", loss, prog_bar=True, batch_size=num_genes_in_batch)
        self.log("val_acc", accuracy, prog_bar=True, batch_size=num_genes_in_batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

    def train_dataloader(self):
        all_train_genes = list(self.train_set.keys())
        return DataLoader(
            PreBatchedDataset(all_train_genes),
            batch_size=self.hparams.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        all_val_genes = list(self.val_set.keys())
        return DataLoader(
            PreBatchedDataset(all_val_genes),
            batch_size=self.hparams.batch_size,
            shuffle=False
        )


def objective(trial):
    # Hyperparameter search space
    target_set_size = trial.suggest_categorical("target_set_size", [400, 600, 800])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2 ** i for i in range(5, 8)])
    temperature = trial.suggest_categorical("temperature", [0.07, 0.1, 0.25])

    # Model
    model = LitMotifBasedEncoder(train_set, val_set, batch_size=batch_size, learning_rate=learning_rate,
                                 temperature=temperature, family_size=8, target_set_size=target_set_size)

    # Trainer
    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        logger=TensorBoardLogger("optuna_logs", name="motif_based_encoder"),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, model.train_dataloader(), model.val_dataloader())

    return trainer.callback_metrics["val_loss"].item()


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best hyperparameters:", study.best_trial.params)
