"""
Dedicated preprocessing utilities for windowed sequence training.

This module provides clean separation between data preparation and model training,
eliminating the need for complex windowing logic inside train_rnn().
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def _left_pad_stack(
    seqs: list[torch.Tensor], *, pad_value: int = 0, target_len: int | None = None
) -> torch.Tensor:
    """
    Left-pad 1D LongTensors to a common length and stack as [B, L].
    
    If target_len is None, pad to the maximum length found in seqs.
    """
    if not seqs:
        return torch.zeros((0, 1), dtype=torch.long)
    max_len = target_len if target_len is not None else max(int(s.numel()) for s in seqs)
    out = torch.full(
        (len(seqs), max_len),
        pad_value,
        dtype=seqs[0].dtype,
        device=seqs[0].device,
    )
    for i, s in enumerate(seqs):
        l = int(s.numel())
        if l == 0:
            continue
        out[i, max_len - l : max_len] = s[-max_len:]
    return out


class WindowedSequenceDataset:
    """
    Create a windowed dataset for recurrent model training.
    
    For each sequence, generates multiple training examples using a sliding window.
    This ensures the model learns to predict from ANY position in the sequence,
    not just the end.
    
    Example:
        For sequence [A, B, C, D, E, F] with window_size=3:
        Creates: [A,B,C], [B,C,D], [C,D,E], [D,E,F]
        
    This matches prefix-based evaluation semantics where predictions are made
    from all possible prefix positions.

    """

    def __init__(
        self,
        sequences: torch.Tensor,
        window_size: int,
        stride: int = 1,
        left_pad: bool = True,
        min_window_size: int = 2,
    ):
        """
        Initialize windowed dataset.

        Args:
            sequences: Input sequences tensor [batch, max_seq_len]
            window_size: Size of sliding window
            stride: Step size for sliding window (default: 1 for maximum overlap)
            left_pad: Whether to left-pad windows shorter than window_size
            min_window_size: Minimum valid window size (default: 2 for input/target split)

        """
        self.window_size = window_size
        self.stride = stride
        self.left_pad = left_pad
        self.min_window_size = min_window_size

        # Generate all windows
        self.windows = self._create_windows(sequences)

    def _create_windows(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Create sliding windows from sequences.

        Returns stacked tensor of windows [num_windows, window_size + 1]
        where +1 accounts for input/target split.
        """
        all_windows = []

        for i in range(sequences.shape[0]):
            seq = sequences[i]
            # Get valid length (exclude padding)
            valid_len = int((seq != 0).sum().item())

            if valid_len < self.min_window_size:
                continue

            # Generate sliding windows
            # We need window_size + 1 tokens (window_size inputs + 1 target)
            max_start = valid_len - self.min_window_size + 1

            for start_pos in range(0, max_start, self.stride):
                # Calculate end position
                # We want window_size inputs, so we need window_size + 1 total tokens
                end_pos = min(start_pos + self.window_size + 1, valid_len + 1)
                window = seq[start_pos:end_pos].clone()

                # Left-pad if window is shorter than window_size + 1
                actual_len = window.numel()
                target_len = self.window_size + 1

                if actual_len < target_len:
                    if self.left_pad:
                        pad_len = target_len - actual_len
                        pad = torch.zeros(pad_len, dtype=window.dtype, device=window.device)
                        window = torch.cat([pad, window])
                    else:
                        # Right-pad (less common for recurrent models)
                        pad_len = target_len - actual_len
                        pad = torch.zeros(pad_len, dtype=window.dtype, device=window.device)
                        window = torch.cat([window, pad])
                elif actual_len > target_len:
                    # Truncate to target length (take last tokens if left-padding)
                    window = window[-target_len:] if self.left_pad else window[:target_len]

                all_windows.append(window)

        if not all_windows:
            # Return empty tensor with correct shape
            return torch.zeros((0, self.window_size + 1), dtype=torch.long)

        # Stack all windows
        return torch.stack(all_windows)

    def __len__(self) -> int:
        """Return number of windows."""
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get window at index."""
        return self.windows[idx]

    def get_tensor(self) -> torch.Tensor:
        """Get all windows as a single tensor."""
        return self.windows


class TransformerPrefixDataset:
    """
    Create prefix-based dataset for Transformer training.

    Generates ALL possible prefixes from each sequence, with optional windowing
    to limit prefix length. This matches the prefix-based evaluation used for
    Transformers where we predict the next token given all previous tokens.

    Example:
        For sequence [A, B, C, D] with window_size=3:
        Creates: [A], [A,B], [A,B,C], [B,C,D]
        
    Unlike sliding windows (which are for RNNs), this creates INCREMENTAL
    prefixes to match how Transformers are evaluated.

    """

    def __init__(
        self,
        sequences: torch.Tensor,
        window_size: int | None = None,
        left_pad: bool = True,
    ):
        """
        Initialize prefix dataset.
        
        Args:
            sequences: Input sequences tensor [batch, max_seq_len]
            window_size: Optional max prefix length (crops longer prefixes)
            left_pad: Whether to left-pad prefixes

        """
        self.window_size = window_size
        self.left_pad = left_pad

        # Generate all prefixes
        self.prefixes = self._create_prefixes(sequences)

    def _create_prefixes(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Create all prefixes from sequences.
        
        Returns stacked tensor of prefixes [num_prefixes, max_prefix_len + 1]
        """
        all_prefixes = []

        for i in range(sequences.shape[0]):
            seq = sequences[i]
            # Split into input and target
            x_seq = seq[:-1]
            y_seq = seq[1:]

            # Get valid length (exclude padding)
            valid_len = int((y_seq != 0).sum().item())

            if valid_len < 1:
                continue

            # Generate incremental prefixes: k=1, k=2, ..., k=valid_len
            for k in range(1, valid_len + 1):
                x_prefix = x_seq[:k].clone()
                y_prefix = y_seq[:k].clone()

                # Combine into single sequence for processing
                prefix = torch.cat([x_prefix, y_prefix[-1:]])  # Input + last target

                # Apply windowing if specified
                if self.window_size is not None and prefix.numel() > self.window_size:
                    prefix = prefix[-self.window_size:]

                all_prefixes.append(prefix)

        if not all_prefixes:
            return torch.zeros((0, 1), dtype=torch.long)

        # Stack with padding
        if self.left_pad:
            max_len = self.window_size if self.window_size else max(p.numel() for p in all_prefixes)
            return _left_pad_stack(all_prefixes, target_len=max_len)
        return pad_sequence(all_prefixes, batch_first=True, padding_value=0)

    def __len__(self) -> int:
        """Return number of prefixes."""
        return self.prefixes.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get prefix at index."""
        return self.prefixes[idx]

    def get_tensor(self) -> torch.Tensor:
        """Get all prefixes as a single tensor."""
        return self.prefixes


def create_windowed_dataset(
    sequences: torch.Tensor,
    window_size: int,
    model_type: str = "lstm",
    stride: int = 1,
    *,
    left_pad: bool = True,
) -> torch.Tensor:
    """
    Create windowed dataset based on model type.

    Args:
        sequences: Input sequences [batch, seq_len]
        window_size: Window size for windowing
        model_type: Either "lstm"/"gru" (sliding windows) or "transformer" (prefixes)
        stride: Stride for sliding windows (LSTM/GRU only)
        left_pad: Whether to left-pad

    Returns:
        Windowed dataset tensor ready for training

    """
    model_type = model_type.lower()

    if model_type in ("lstm", "gru", "rnn"):
        # Use sliding windows for recurrent models
        dataset = WindowedSequenceDataset(
            sequences=sequences,
            window_size=window_size,
            stride=stride,
            left_pad=left_pad,
        )
    elif model_type == "transformer":
        # Use prefix-based dataset for Transformers
        dataset = TransformerPrefixDataset(
            sequences=sequences,
            window_size=window_size,
            left_pad=left_pad,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'lstm', 'gru', or 'transformer'")

    return dataset.get_tensor()


# Example usage and validation
if __name__ == "__main__":
    # Test with sample sequences
    test_sequences = torch.tensor([
        [1, 2, 3, 4, 5, 6, 0, 0],  # Valid length: 6
        [7, 8, 9, 10, 0, 0, 0, 0],  # Valid length: 4
        [11, 12, 13, 14, 15, 0, 0, 0],  # Valid length: 5
    ])

    print("Original sequences:")
    print(test_sequences)
    print()

    # Test LSTM windowing
    print("LSTM sliding windows (window_size=3):")
    lstm_windows = create_windowed_dataset(
        test_sequences, window_size=3, model_type="lstm"
    )
    print(f"Shape: {lstm_windows.shape}")
    print(lstm_windows[:5])
    print()

    # Test Transformer prefixes
    print("Transformer prefixes (window_size=3):")
    transformer_prefixes = create_windowed_dataset(
        test_sequences, window_size=3, model_type="transformer"
    )
    print(f"Shape: {transformer_prefixes.shape}")
    print(transformer_prefixes[:5])
    print()

    # Verify alignment for sequence [1,2,3,4,5,6]:
    # LSTM should create: [1,2,3,4], [2,3,4,5], [3,4,5,6]
    # Transformer should create: [1], [1,2], [1,2,3], [2,3,4], [3,4,5], [4,5,6]
    print("Expected LSTM windows from [1,2,3,4,5,6]:")
    print("  [0,1,2,3,4], [0,2,3,4,5], [0,3,4,5,6]  (left-padded to size 4)")
    print()
    print("Expected Transformer prefixes from [1,2,3,4,5,6]:")
    print("  [0,0,1,2], [0,1,2,3], [1,2,3,4], [2,3,4,5], [3,4,5,6]")
