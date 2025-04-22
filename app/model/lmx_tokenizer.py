from ..olimpic_icdar24.app.linearization.vocabulary import ALL_TOKENS


class LMXTokenizer:
    """Tokenizer for LMX sequences handling special tokens and vocabulary mapping.

    Converts between LMX token sequences and numerical IDs, handling special tokens
    for padding, start/end of sequence, and unknown tokens.
    """

    def __init__(self) -> None:
        """Initialize tokenizer with merged vocabulary of special and domain tokens.

        Vocabulary construction:
        - Special tokens occupy first 4 indices
        - Domain tokens from ALL_TOKENS start at index 4
        - Maintains bidirectional lookup dictionaries
        """
        lmx_vocab = {
            token: idx for idx, token in enumerate(ALL_TOKENS, start=4)
        }  # Start from 4 to leave space for special tokens

        # Add special tokens
        special_tokens = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.vocab = {
            **special_tokens,
            **lmx_vocab,
        }  # Merge special tokens with the vocabulary

        # Reverse lookup for decoding
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def encode(self, lmx_sequence: str) -> list[int]:
        """Convert space-separated LMX tokens to numerical IDs.

        Args:
            lmx_sequence (str): Space-delimited sequence of LMX tokens

        Returns:
            list[int]: List of token IDs where unknown tokens are mapped to <unk>
        """
        return [
            self.vocab.get(token, self.vocab["<unk>"]) for token in lmx_sequence.split()
        ]

    def decode(self, token_ids: list[int]) -> str:
        """Convert numerical IDs back to LMX tokens.

        Args:
            token_ids (list[int]): Sequence of token IDs to convert

        Returns:
            str: Space-delimited LMX token sequence with <unk> for invalid IDs
        """
        return " ".join([self.reverse_vocab.get(idx, "<unk>") for idx in token_ids])
