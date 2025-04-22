from pathlib import Path
from typing import Any

import torch
import datetime
import re

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning import Trainer

from .model import config
from .model.config import *
from .model.lmx_tokenizer import LMXTokenizer
from .model.lmx_data_module import LMXDataModule
from .model.transformer_model import TransformerModel
from .music_data_preprocessor import MusicDataPreprocessor

BASE_DIR = Path(__file__).parent

class LMXGenerator:
    """Main interface for LMX sequence generation and model_name training.

    Handles end-to-end pipeline including:
    - Model initialization
    - Sequence generation
    - Training workflow
    - Checkpoint management
    """

    def __init__(
        self, load_pretrained_model: bool = True, model_name: str | None = None
    ) -> None:
        """Initialize LMX generator with model_name components.

        Args:
            load_pretrained_model (bool): Whether to load existing weights
            model_name (str, optional): Model filename in trained_models directory
        """
        self.tokenizer: LMXTokenizer = LMXTokenizer()
        self.max_length = MAX_LENGTH
        self.model = TransformerModel(
            tokenizer=self.tokenizer,
            learning_rate=LEARNING_RATE,
            max_length=MAX_LENGTH,
            embed_size=EMBED_SIZE,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT_RATE,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model is running on {self.device}")

        if load_pretrained_model:
            self._load_model(model_name)

    def _load_model(self, model_name: str) -> None:
        """Load pretrained model_name weights from checkpoint.

        Args:
            model_name (str): Model filename in trained_models directory
        """
        if not model_name:
            model_name = "lmx_model.pt"
        model_filepath = BASE_DIR / "trained_models" / model_name
        if not model_filepath.exists():
            raise FileNotFoundError(f"Model {model_name} not found at {model_filepath}")
        checkpoint = torch.load(model_filepath, map_location=self.device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)

    def generate(
        self,
        seed_sequence: str | None = None,
        temperature=1,
        top_k=0,
        num_samples=1,
        max_new_tokens: int = 1000,
    ):
        """Generate LMX sequences from a seed using autoregressive prediction.

        Args:
            seed_sequence (str, optional): Starting sequence for generation
            temperature (float): Sampling temperature (1.0 = neutral)
            top_k (int): Top-k filtering threshold (0 = disabled)
            num_samples (int): Number of independent sequences to generate
            max_new_tokens (int): Maximum tokens to generate per sequence

        Returns:
            str | list[str]: Generated sequence(s) with special tokens removed
        """
        self.model.eval()
        self.model.tokenizer = self.tokenizer  # Give model_name access to tokenizer

        # Process seed lmx_sequence
        if not seed_sequence:
            seed_sequence = "<sos>"
        elif not seed_sequence.startswith("<sos>"):
            seed_sequence = "<sos> " + seed_sequence

        # Encode the seed
        token_ids = torch.tensor(self.tokenizer.encode(seed_sequence))
        token_ids_batch = token_ids.unsqueeze(0).expand(num_samples, -1).to(self.device)

        with torch.no_grad():
            # Generate lmx_sequence
            generated_seqs = self.model.predict(
                token_ids_batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        results = []
        for generated_seq in generated_seqs.tolist():
            # Decode the generated token IDs
            generated_seq = self.tokenizer.decode(generated_seq[1:])  # Skip <sos>

            # Remove <eos> and everything after
            if "<eos>" in generated_seq:
                generated_seq = generated_seq.split("<eos>")[0].strip()

            results.append(generated_seq)

        return results[0] if num_samples == 1 else results

    def _create_logger(self) -> TensorBoardLogger:
        """Configure TensorBoard logger with hyperparameter tracking.

        Returns:
            TensorBoardLogger: Configured logger instance
        """
        hyperparam_vars: dict[str, Any] = {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("__")
        }
        logdir = Path(
            "lightning_logs",
            "{}-{}".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
                ",".join(
                    (
                        "{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                        for k, v in sorted(hyperparam_vars.items())
                    )
                ),
            ),
        )
        return TensorBoardLogger(logdir)

    def train(self, dataset_filename: str, checkpoint_path_to_load: str | None = None, use_logger: bool = False):
        """Execute full training workflow with validation and testing.

        Args:
            dataset_filename (str): Path to preprocessed dataset pickle file inside converted_dataset folder
            checkpoint_path_to_load (str, optional): Absolute path to resume training from
            use_logger (bool): Set if logger should be used (needs TensorBoard installed)
        """
        logger = self._create_logger() if use_logger else None
        lmx_data = MusicDataPreprocessor().load_from_pickle(dataset_filename)
        data_module = LMXDataModule(
            lmx_data, self.tokenizer, batch_size=BATCH_SIZE, max_length=self.max_length
        )

        trainer = Trainer(
            logger=logger,
            max_epochs=MAX_EPOCHS,
            accelerator=self.device,
            devices="auto",
            gradient_clip_val=1.0,
            enable_checkpointing=True,
            log_every_n_steps=20,
            num_sanity_val_steps=1,
            callbacks=[
                EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10),
                ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min"),
            ],
        )
        trainer.fit(
            self.model, datamodule=data_module, ckpt_path=checkpoint_path_to_load
        )

        trainer.test(self.model, datamodule=data_module)
        torch.save(self.model.state_dict(), BASE_DIR / "trained_models" / "new_lmx_model.pt")
