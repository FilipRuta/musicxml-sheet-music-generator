import torch

from torch.utils.data import Dataset

from .lmx_tokenizer import LMXTokenizer


class LMXDataset(Dataset):
    """Dataset for handling LMX sequence tokenization and preprocessing.

    Converts raw LMX sequences into tokenized tensors with proper padding/truncation,
    and prepares input-target pairs for sequence model_name training.
    """

    def __init__(
        self, lmx_data: list[str], tokenizer: LMXTokenizer, max_length: int
    ) -> None:
        """
        Args:
            lmx_data (list[str]): Raw LMX sequences to process
            tokenizer (LMXTokenizer): Tokenizer instance with vocabulary mapping
            max_length (int): Maximum sequence length including special tokens
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data: list[torch.tensor] = []

        # Tokenize and encode LMX sequences
        for lmx in lmx_data:
            # lmx = "measure".join(lmx.split("measure")[:3])  # include only n - 1 measures
            token_ids = self.tokenizer.encode(lmx)
            # Pad or truncate to the desired length
            token_ids = self.pad_or_truncate(token_ids)
            self.data.append(torch.tensor(token_ids, dtype=torch.long))

    def pad_or_truncate(self, token_ids: list[int]) -> list[int]:
        """Process token sequences to fixed length with special tokens.

        Args:
            token_ids (list[int]): Raw token IDs from tokenizer

        Returns:
            list[int]: Processed tokens with <sos>/<eos> and padding

        Processing steps:
        1. Truncate to max_length - 1 (reserving space for <sos>)
        2. Add <sos> token at beginning
        3. Add <eos> token and padding tokens at end
        """
        max_len = self.max_length - 1  # Reserve space for <sos>

        # Truncate if too long
        token_ids = token_ids[:max_len]

        # Add <sos> and <eos>
        token_ids = [self.tokenizer.vocab["<sos>"]] + token_ids

        # Pad if too short
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids += [self.tokenizer.vocab["<eos>"]] + [
                self.tokenizer.vocab["<pad>"]
            ] * (padding_length - 1)

        return token_ids

    def __len__(self) -> int:
        """Get total number of processed sequences.

        Returns:
            int: Count of available samples
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get input-target pair for model_name training.

        Args:
            idx (int): Index of the sequence to retrieve

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (input_ids, target_ids) pair where:
            - input_ids: Sequence tokens from start to second-last position
            - target_ids: Sequence tokens from second position to end
        """
        token_ids = self.data[idx]
        input_ids = token_ids[:-1]  # Exclude the last token for input
        target_ids = token_ids[1:]  # Shifted by 1 for target
        return input_ids, target_ids
