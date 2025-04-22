import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR

from .lmx_tokenizer import LMXTokenizer


class TransformerModel(pl.LightningModule):
    """Transformer-based model_name for LMX sequence generation using decoder-only architecture.

    Implements a GPT-style transformer with positional encoding, self-attention layers,
    and autoregressive generation capabilities.
    """

    def __init__(
        self,
        tokenizer: LMXTokenizer,
        learning_rate=1e-3,
        embed_size=256,
        num_heads=8,
        num_layers=6,
        ff_hidden_size=512,
        max_length=512,
        dropout=0.1,
    ) -> None:
        """
        Args:
            tokenizer (LMXTokenizer): Tokenizer for vocabulary conversions
            learning_rate (float): Initial learning rate for optimizer
            embed_size (int): Token embedding dimensionality
            num_heads (int): Number of attention heads per layer
            num_layers (int): Number of transformer layers
            ff_hidden_size (int): Feed-forward network hidden dimension
            max_length (int): Maximum sequence length for positional encoding
            dropout (float): Dropout probability for regularization
        """
        super(TransformerModel, self).__init__()
        self.lr = learning_rate
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.vocab)
        self.max_length = max_length
        self.random_gen = torch.Generator(device=self.device).manual_seed(42)
        # Embedding layer for tokens
        self.embedding = nn.Embedding(self.vocab_size, embed_size)

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            self._generate_positional_encoding(max_length, embed_size),
            requires_grad=False,
        )

        # Transformer for decoder-only architecture (using self attention blocks only)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=ff_hidden_size,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            bias=False,
        )
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Output projection (convert embeddings back to token IDs)
        self.fc_out = nn.Linear(embed_size, self.vocab_size)

        # Loss function (ignoring padding tokens)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # <pad> token index

    def _generate_positional_encoding(
        self, max_length: int, embed_size: int
    ) -> torch.Tensor:
        """Generate fixed sinusoidal positional encodings.

        Args:
            max_length (int): Maximum sequence length
            embed_size (int): Embedding dimension size

        Returns:
            torch.Tensor: Positional encoding matrix of shape (max_length, embed_size)
        """
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2)
            * -(torch.log(torch.tensor(10000.0)) / embed_size)
        )
        pe = torch.zeros(max_length, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Model forward pass with causal masking.

        Args:
            x (torch.Tensor): Input token IDs of shape (batch_size, seq_len)

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        seq_len = x.size(1)

        # Create mask for hiding the future tokens
        causal_mask = (
            torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
            .bool()
            .to(x.device)
        )
        embedded = self.embedding(x)

        # Add positional encoding to token embeddings
        x = embedded + self.positional_encoding[: x.size(1), :]

        # No need to transpose, EncoderLayers have batch_first=True so it expects current format batch, seq_len, embed
        # Pass through Transformer Encoder
        x = self.transformer(x, mask=causal_mask)
        logits = self.fc_out(x)
        return logits

    def common_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Shared logic for training/validation/test steps.

        Args:
            batch (tuple): (input_ids, target_ids) tensors
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Computed loss value
        """
        input_ids, target_ids = batch  # -> batch, seq_len
        output = self(
            input_ids
        )  # Model forward pass, output: batch, seq_len, vocab_size
        B, T, C = output.shape
        output = output.view(B * T, C)
        target_ids = target_ids.view(B * T)
        loss = self.loss_fn(output, target_ids)  # Calculate loss
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step"""
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step with perplexity and generation quality checks."""
        input_ids, target_ids = batch
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)

        perplexity = torch.exp(loss)
        self.log("val_perplexity", perplexity, prog_bar=True)

        # For a small subset of batches, test generation
        if batch_idx % 20 == 0 and batch_idx > 0:  # Every 20 batches
            # Take first lmx_sequence from batch for sampling
            seed = input_ids[0, :50]  # Use first 50 tokens as seed

            # Generate continuation
            with torch.no_grad():
                generated = self.predict(seed.unsqueeze(0), max_new_tokens=50)

            # Compare generated lmx_sequence with actual targets
            target_seq = target_ids[0, 50:100]  # Next 50 tokens
            gen_seq = generated[0, 50:100]

            # Calculate token accuracy
            correct = (gen_seq == target_seq).sum().item()
            total = len(target_seq)
            accuracy = correct / total

            self.log("val_generation_accuracy", accuracy)

        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Test step"""
        loss = self.common_step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def predict(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 500,
        temperature: float = 0.5,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Autoregressive sequence generation with temperature and top-k sampling.

        Args:
            input_ids (torch.Tensor): Seed sequence tensor of shape (batch_size, seq_len)
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature (higher = more random)
            top_k (int): If >0, use top-k filtering (0 = no filtering)

        Returns:
            torch.Tensor: Generated sequence tensor of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        generated = input_ids.clone()  # batch, seq_len

        # Prepare batch sized tensor to track which sequences finished already
        eos_generated = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        eos_token = self.tokenizer.vocab.get("<eos>", -1)

        with torch.no_grad():
            for _ in range(max_new_tokens):

                # Get predictions
                outputs = self(
                    generated[:, -min(generated.shape[1], self.max_length) :]
                )
                next_token_logits = outputs[:, -1, :].squeeze(1)

                # Apply temperature
                scaled_logits = next_token_logits / temperature

                if top_k > 0:
                    # Keep only top-k tokens
                    top_k_values, top_k_indices = torch.topk(
                        scaled_logits, top_k, dim=-1
                    )

                    # Create a mask of invalid tokens
                    invalid_tokens = torch.ones_like(scaled_logits).bool()
                    invalid_tokens.scatter_(1, top_k_indices, False)

                    # Set invalid tokens to -inf
                    scaled_logits.masked_fill_(invalid_tokens, float("-inf"))

                # Apply softmax to get probabilities
                probs = torch.softmax(scaled_logits, dim=-1)

                # Sample from the distribution
                next_token = torch.multinomial(probs, 1)

                # Append new token
                generated = torch.cat([generated, next_token], dim=1)

                # Update info about ended sequences
                new_eos_indices = next_token.squeeze(1) == eos_token
                eos_generated = eos_generated | new_eos_indices

                # Stop if all sequences have generated EOS
                if torch.all(eos_generated):
                    break

        return generated

    def configure_optimizers(self) -> tuple[list, list]:
        """
        Configure optimizers for training.
        """
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=0.01
        )

        def lr_lambda(current_step):
            warmup_steps = 500
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0  # keep LR constant after warmup (you can change this)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer, T_0=10
        # )
        return [self.optimizer], [{"scheduler": LambdaLR(self.optimizer, lr_lambda), "interval": "step", "frequency": 1}]
