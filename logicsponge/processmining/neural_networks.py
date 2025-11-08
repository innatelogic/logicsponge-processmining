"""The NN architectures."""

import copy
import logging
import time

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from logicsponge.processmining.encodings import (
    BackwardRelativePositionalEncoding,
    LearnableRelativePositionalEncoding,
    PeriodicPositionalEncoding,
    SharpPeriodicRelativeEncoding,
    SinusoidalPositionalEncoding,
)
from logicsponge.processmining.types import ActivityName, Event

logger = logging.getLogger(__name__)


# ============================================================
# Models (RNN and LSTM)
# ============================================================


class RNNModel(nn.Module):
    """
    Simple RNN model for sequence prediction.

    This model uses two RNN layers to process sequences of activities.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embeddings.
        hidden_dim (int): Dimension of the hidden layers in the RNN.
        output_dim (int): Dimension of the output layer.
        device (torch.device | None): Device to run the model on (CPU or GPU).

    """

    device: torch.device | None
    embedding: nn.Embedding
    rnn1: nn.RNN
    rnn2: nn.RNN
    fc: nn.Linear

    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, device: torch.device | None = None
    ) -> None:
        """
        Initialize the RNN model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the hidden layers in the RNN.
            output_dim (int): Dimension of the output layer.
            device (torch.device | None): Device to run the model on (CPU or GPU).

        """
        super().__init__()
        self.device = device
        # Use padding_idx=0 to handle padding, same as in LSTMModel
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0, device=device)

        # Two RNN layers, similar to the LSTMModel
        self.rnn1 = nn.RNN(embedding_dim, hidden_dim, batch_first=True, device=device)
        self.rnn2 = nn.RNN(hidden_dim, hidden_dim, batch_first=True, device=device)

        # Fully connected layer to predict next activity
        self.fc = nn.Linear(hidden_dim, output_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len), where each element is an activity index.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim),
            where each element is the predicted activity.

        """
        # Convert activity indices to embeddings
        x = self.embedding(x)

        # Pass through layers
        rnn_out, _ = self.rnn1(x)
        rnn_out, _ = self.rnn2(rnn_out)

        return self.fc(rnn_out)


class LSTMModel(nn.Module):
    """
    LSTM model for sequence prediction.

    This model uses two LSTM layers to process sequences of activities.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embeddings.
        hidden_dim (int): Dimension of the hidden layers in the LSTM.
        output_dim (int): Dimension of the output layer.
        use_one_hot (bool): Whether to use one-hot encoding instead of embeddings.
        device (torch.device | None): Device to run the model on (CPU or GPU).

    Methods:
    - forward(x, hidden=None) -> (logits, new_hidden) where logits shape is [B, L, V]
    - step(input_token, hidden=None) -> (logits_last_step, new_hidden) where logits_last_step shape is [B, V]

    """

    device: torch.device | None
    embedding: nn.Embedding | None
    use_one_hot: bool
    vocab_size: int
    embedding_dim: int
    lstm1: nn.LSTM
    lstm2: nn.LSTM
    fc: nn.Linear

    def __init__( # noqa: PLR0913
        self,
        vocab_size: int,
        *,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        padding_idx: int = 0,
        use_one_hot: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the LSTM model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the hidden layers in the LSTM.
            num_layers (int): Number of LSTM layers.
            padding_idx (int): Padding index for the embedding layer.
            use_one_hot (bool): Whether to use one-hot encoding instead of embeddings.
            device (torch.device | None): Device to run the model on (CPU or GPU).

        """
        super().__init__()
        self.device = device
        self.use_one_hot = use_one_hot
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Conditional embedding layer
        if not use_one_hot:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx, device=device)
        else:
            self.embedding = None

        # Input dimension to LSTM depends on the encoding method
        input_dim = vocab_size if use_one_hot else embedding_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, device=device)
        self.fc = nn.Linear(hidden_dim, vocab_size, device=device)

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def forward(
            self,
            x: torch.Tensor,
            hidden: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len), where each element is an activity index.
            hidden (tuple[torch.Tensor, torch.Tensor] | None): Optional initial hidden and cell states for the LSTM.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim),
            where each element is the predicted activity.

        """
        if not self.use_one_hot and self.embedding is not None:
            # Use embedding layer
            x = self.embedding(x)
        else:
            # Use one-hot encoding
            # print(f"x shape: {x.shape}, dtype: {x.dtype}, unique values: {torch.unique(x)}")
            # Keep device consistent with input tensor to avoid device mismatch
            x = torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float().to(x.device)

        # Pass through LSTM layers
        lstm_out, new_hidden = self.lstm(x, hidden)
        logits = self.fc(lstm_out)

        return logits, new_hidden

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)
        elif isinstance(m, nn.Embedding) and m is not None:
            nn.init.uniform_(m.weight, -0.1, 0.1)


class GRUModel(nn.Module):
    """
    GRU model for sequence prediction (LSTM-like API).

    Supports either an embedding lookup or one-hot input representation.

    Methods:
    - forward(x, hidden=None) -> (logits, new_hidden) where logits shape is [B, L, V]
    - step(input_token, hidden=None) -> (logits_last_step, new_hidden) where logits_last_step shape is [B, V]

    """

    device: torch.device | None
    embedding: nn.Embedding | None
    use_one_hot: bool
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    gru: nn.GRU
    fc: nn.Linear

    def __init__(  # noqa: PLR0913
        self,
        vocab_size: int,
        *,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        padding_idx: int = 0,
        use_one_hot: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the GRU model with optional one-hot input representation."""
        super().__init__()
        self.device = device
        self.use_one_hot = use_one_hot
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Conditional embedding layer
        if not use_one_hot:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx, device=device)
        else:
            self.embedding = None

        # Input dim depends on representation
        input_dim = vocab_size if use_one_hot else embedding_dim

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, device=device)
        self.fc = nn.Linear(hidden_dim, vocab_size, device=device)

        self.apply(self._init_weights)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GRU.

        Args:
            x: [batch, seq_len] int64 token indices
            hidden: optional hidden state

        Returns:
            logits: [batch, seq_len, vocab_size]
            new_hidden: GRU hidden state

        """
        if not self.use_one_hot and self.embedding is not None:
            x = self.embedding(x)
        else:
            x = F.one_hot(x, num_classes=self.vocab_size).float().to(x.device)

        gru_out, new_hidden = self.gru(x, hidden)
        logits = self.fc(gru_out)
        return logits, new_hidden

    def step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-timestep update compatible with streaming caches.

        Args:
            input_token: [batch] or [batch, 1] indices
            hidden: prior hidden state

        Returns:
            logits_last: [batch, vocab_size]
            new_hidden: updated hidden

        """
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)

        if not self.use_one_hot and self.embedding is not None:
            x = self.embedding(input_token)
        else:
            x = F.one_hot(input_token, num_classes=self.vocab_size).float().to(input_token.device)

        gru_out, new_hidden = self.gru(x, hidden)
        logits_last = self.fc(gru_out[:, -1, :])
        return logits_last, new_hidden

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)
        elif isinstance(m, nn.Embedding) and m is not None:
            nn.init.uniform_(m.weight, -0.1, 0.1)


class TransformerModel(nn.Module):
    """
    Transformer model for sequence prediction.

    This model uses a transformer encoder to process sequences of activities.
    It can handle both one-hot encoded inputs and embeddings.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embeddings.
        hidden_dim (int): Dimension of the hidden layers in the transformer.
        output_dim (int): Dimension of the output layer.
        use_one_hot (bool): Whether to use one-hot encoding instead of embeddings.
        device (torch.device | None): Device to run the model on (CPU or GPU).

    """

    device: torch.device | None
    embedding: nn.Embedding | None
    use_one_hot: bool
    vocab_size: int
    embedding_dim: int
    # positional encodings are computed on-the-fly (sinusoidal), no learned parameter
    # so no fixed bound on sequence length
    transformer: nn.TransformerEncoder
    fc: nn.Linear

    def __init__( # noqa: PLR0913
        self,
        seq_input_dim: int,  # Interpreted as max_seq_len for positional encodings
        vocab_size: int,
        *,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        padding_idx: int = 0,
        output_dim: int = 64,
        attention_heads: int = 4,
        use_one_hot: bool = False,
        device: torch.device | None = None,
        pos_encoding_type: str = "learnable_backward_relative",
        sharp_mode: str = "square",
    ) -> None:
        """
        Initialize the Transformer model.

        Args:
            seq_input_dim (int): Input dimension for sequences.
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the hidden layers in the transformer.
            num_layers (int): Number of transformer layers.
            padding_idx (int): Padding index for the embedding layer.
            output_dim (int): Dimension of the output layer.
            attention_heads (int): Number of attention heads in the transformer.
            use_one_hot (bool): Whether to use one-hot encoding instead of embeddings.
            device (torch.device | None): Device to run the model on (CPU or GPU).
            pos_encoding_type (str): Type of positional encoding to use.
                Options: "sinusoidal", "learnable_backward_relative", "periodic", "sharp_relative", "learnable_relative"
            sharp_mode (str): Mode for sharp periodic relative encoding. Options: "square", "sawtooth", "quantized".

        """
        super().__init__()
        self.device = device
        self.use_one_hot = use_one_hot
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Model dimension for the Transformer (d_model) depends on representation
        # - If embeddings are used, d_model = embedding_dim
        # - If one-hot is used, d_model = vocab_size
        d_model = embedding_dim if not use_one_hot else vocab_size
        self.d_model = d_model

        # Use a simple, reliable learned positional embedding up to `seq_input_dim`.
        # This is robust and easy to reason about. If an input sequence is longer
        # than `seq_input_dim`, we'll fall back to a sinusoidal encoding at runtime.
        self.max_seq_len = int(seq_input_dim)
        self.pos_embedding = nn.Embedding(self.max_seq_len, d_model, device=device)
        # Conditional embedding layer
        if not use_one_hot:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx, device=device)
        else:
            self.embedding = None

        # Use unbounded sinusoidal positional encoding computed on-the-fly; no learned parameter

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
            device=device,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer -- produce logits over the vocabulary (vocab_size)
        # Training/evaluation code expects shape [batch, seq_len, vocab_size]
        self.fc = nn.Linear(d_model, vocab_size, device=device)

        # Custom initialization for linear/embedding layers
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len), where each element is an activity index.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim),
            where each element is the predicted activity.

        """
        # Ensure we keep computations on the same device as input indices
        x_device = x.device
        # Create padding mask (True for padding positions) from token indices
        key_padding_mask = x == 0

        if not self.use_one_hot and self.embedding is not None:
            x = self.embedding(x)
        else:
            # Project indices to one-hot vectors on the same device as input
            x = F.one_hot(x, num_classes=self.vocab_size).float().to(x_device)

        # Add positional embeddings. If sequence length exceeds learned table,
        # fall back to a sinusoidal positional encoding.
        batch_size, seq_len, _ = x.size()

        if seq_len <= self.max_seq_len:
            pos_ids = torch.arange(seq_len, device=x_device).unsqueeze(0).expand(batch_size, seq_len)
            pos_emb = self.pos_embedding(pos_ids)
        else:
            # Fallback: use sinusoidal positional encodings computed on the fly
            # The SinusoidalPositionalEncoding implementation (imported) typically
            # returns shape [1, seq_len, d_model] when called as (seq_len, device, dtype)
            try:
                pos_emb = SinusoidalPositionalEncoding(self.d_model)(seq_len, x_device, x.dtype)
                if pos_emb.dim() == 3 and pos_emb.size(0) == 1:
                    pos_emb = pos_emb.expand(batch_size, -1, -1)
                else:
                    pos_emb = pos_emb.to(x_device).expand(batch_size, -1, -1)
            except Exception:
                # last-resort: zeros
                pos_emb = torch.zeros((batch_size, seq_len, self.d_model), device=x_device, dtype=x.dtype)

        x = x + pos_emb

        # === Add causal (left) mask ===
        # Shape: [seq_len, seq_len]; True where positions should be masked (prevent attending)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x_device), diagonal=1).bool()

        # Pass through transformer with mask and key padding mask
        x = self.transformer(x, mask=mask, src_key_padding_mask=key_padding_mask.to(x_device))

        # Project to vocabulary logits
        return self.fc(x)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding) and m is not None:
            nn.init.uniform_(m.weight, -0.1, 0.1)
        # No learned positional parameter; nothing extra to init here


# Small Q-network compatible with sequence input (embedding + GRU + linear)
class QNetwork(nn.Module):
    """
    Simple Q-network with embedding, GRU, and linear layers.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embeddings.
        hidden_dim (int): Dimension of the hidden layer in the GRU.
        output_dim (int): Dimension of the output layer.
        device (torch.device | None): Device to run the model on (CPU or GPU).

    """

    def __init__(  # noqa: PLR0913
        self,
        vocab_size: int,
        *,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        padding_idx: int = 0,
        device: torch.device | None = None,
        use_one_hot: bool = False,
        model_architecture: str = "gru"
    ) -> None:
        """
        Initialize the Q-network.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings (feature size per timestep when embeddings are used).
            hidden_dim (int): Dimension of the hidden layer in the GRU.
            num_layers (int): Number of GRU layers (currently unused, single layer GRU is used).
            padding_idx (int): Padding index for the embedding layer.
            device (torch.device | None): Device to run the model on (CPU or GPU).
            use_one_hot (bool): Whether to use one-hot encoding instead of
                embeddings. If True, the GRU input size is set to
                `vocab_size`.
            model_architecture (str): Which architecture to use for the Q-network.
                Supported values: "gru" (default) or "linear". "linear" applies a
                per-timestep linear projection without any recurrence.

        """
        super().__init__()
        self.device = device
        # Optionally disable embedding and process raw one-hot inputs directly.
        self.use_one_hot = use_one_hot
        self.vocab_size = vocab_size
        self.model_architecture = (model_architecture or "gru").lower()

        if not self.use_one_hot:
            # simple embedding (use embedding indices like other NN models)
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx, device=device)
        else:
            # when using raw one-hot inputs, we don't create an embedding layer
            self.embedding = None

        # Input dim depends on whether we use embedding or one-hot
        in_dim = vocab_size if self.use_one_hot else embedding_dim

        if self.model_architecture == "gru":
            # Recurrent architecture (default)
            self.gru = nn.GRU(in_dim, hidden_dim, num_layers=num_layers, batch_first=True, device=device)
            self.fc = nn.Linear(hidden_dim, vocab_size, device=device)
        elif self.model_architecture == "linear":
            # 3-layer feed-forward MLP applied per timestep independently (no recurrence)
            # Shape in: [..., in_dim] -> hidden_dim -> hidden_dim -> vocab_size
            self.gru = None
            self.fc = nn.Sequential(
                nn.Linear(in_dim, hidden_dim, device=device),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, device=device),
                nn.ReLU(),
                nn.Linear(hidden_dim, vocab_size, device=device),
            )
        else:
            msg = f"Unsupported model_architecture '{model_architecture}'. Use 'gru' or 'linear'."
            raise ValueError(msg)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def forward(
            self,
            x: torch.Tensor,
            hidden: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the Q-network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len), where each element is an activity index.
            hidden (torch.Tensor | None): Hidden state tensor from the previous timestep.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, output_dim),
            where each element is the predicted Q-value for the next activity.

        """
        # x: LongTensor [batch, seq_len]
        if not self.use_one_hot and self.embedding is not None:
            emb = self.embedding(x)  # [batch, seq_len, embedding_dim]
        else:
            # Convert indices to one-hot; padding index 0 -> all-zeros vector
            emb = F.one_hot(x, num_classes=self.vocab_size).float().to(x.device)

        if self.model_architecture == "gru":
            if self.gru is None:
                msg = "GRU module is not initialized for architecture 'gru'"
                raise RuntimeError(msg)
            gru_output, new_hidden = self.gru(emb, hidden)  # out: [batch, seq_len, hidden_dim]
            logits = self.fc(gru_output)
            return logits, new_hidden
        # linear: project each timestep independently
        logits = self.fc(emb)
        return logits, None

    def step(
            self,
            input_token: torch.Tensor,
            hidden: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Step through the GRU with a single input token.

        Args:
            input_token (torch.LongTensor): Input tensor of shape (batch_size,) or (batch_size, 1),
                where each element is an activity index.
            hidden (torch.Tensor | None): Hidden state tensor from the previous timestep.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim),
            where each element is the predicted Q-value for the next activity.

        """
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)

        if not self.use_one_hot and self.embedding is not None:
            emb = self.embedding(input_token)  # [batch, 1, embedding_dim]
        else:
            # Convert indices to one-hot; padding index 0 -> all-zeros vector
            emb = F.one_hot(input_token, num_classes=self.vocab_size).float().to(input_token.device)

        if self.model_architecture == "gru":
            if self.gru is None:
                msg = "GRU module is not initialized for architecture 'gru'"
                raise RuntimeError(msg)
            gru_output, new_hidden = self.gru(emb, hidden)
            logits = self.fc(gru_output[:, -1, :])
            return logits, new_hidden
        # linear: single-timestep projection
        logits = self.fc(emb[:, -1, :])
        return logits, None

# ============================================================
# Training and Evaluation
# ============================================================


class PreprocessData:
    """
    Preprocesses sequences of events for training RNN models.

    This class handles the conversion of activity names to indices, which can be used
    as input to the embedding layer of the RNN model. It also pads sequences to ensure
    consistent input shapes.

    Attributes:
        activity_to_idx (dict): Maps activity names to unique indices.
        idx_to_activity (dict): Maps indices back to activity names.
        current_idx (int): Current index for the next activity to be added.

    """

    def __init__(self) -> None:
        """
        Initialize the PreprocessData class.

        Sets up dictionaries for mapping activities to indices and vice versa.
        Initializes the current index to 1 (0 is reserved for padding).
        """
        self.activity_to_idx = {}
        self.idx_to_activity = {}
        self.current_idx = 1  # 0 is reserved for padding

    # Function to get the activity index (for the embedding layer)
    def get_activity_index(self, activity: ActivityName) -> int:
        """
        Get the index for a given activity name, adding it to the mapping if it doesn't exist.

        Args:
            activity (ActivityName): The name of the activity.

        Returns:
            int: The index of the activity, ensuring it is unique.

        """
        if activity not in self.activity_to_idx:
            self.activity_to_idx[activity] = self.current_idx
            self.idx_to_activity[self.current_idx] = activity
            self.current_idx += 1
        return self.activity_to_idx[activity]

    def preprocess_data(self, dataset: list[list[Event]]) -> torch.Tensor:
        """
        Preprocess the dataset of sequences of events.

        Converts activity names to indices and pads sequences to ensure consistent input shapes.

        Args:
            dataset (list[list[Event]]): A list of sequences, where each sequence is a list of events.
            Each event is expected to have an "activity" key with the activity name.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, max_seq_len) containing the indices of activities,
            padded to the maximum sequence length in the dataset. Padding is done with 0.

        """
        processed_sequences = []

        for sequence in dataset:
            index_sequence = [self.get_activity_index(event["activity"]) for event in sequence]  # Convert to indices
            processed_sequences.append(torch.tensor(index_sequence, dtype=torch.long))

        # Pad sequences, using 0 as the padding value
        return pad_sequence(processed_sequences, batch_first=True, padding_value=0)


# === RL training/evaluation helpers for QNetwork (batch mode) ===
RL_MIN_PREFIX_LEN = 1

def _sample_prefix_batch(batch_sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """

    Given a batch of padded sequences [B, L], sample one prefix per row and its next-token target.

    Returns:
        x_inputs: padded LongTensor [B_eff, L_max] of prefixes (padding=0)
        y_targets: LongTensor [B_eff] of next tokens for each sampled prefix

    Skips sequences shorter than RL_MIN_PREFIX_LEN.

    """
    device = batch_sequences.device
    bsz, _ = batch_sequences.shape
    inputs: list[torch.Tensor] = []
    targets: list[int] = []

    # compute effective lengths (exclude padding=0)
    lengths = (batch_sequences != 0).sum(dim=1).tolist()

    for b in range(bsz):
        length = lengths[b]
        if length is None or length < RL_MIN_PREFIX_LEN:
            continue
        # sample a cut point k in [1, length-1]
        k = int(torch.randint(1, length, (1,), device=device).item())
        prefix = batch_sequences[b, :k]  # shape [k]
        target = int(batch_sequences[b, k].item())
        inputs.append(prefix)
        targets.append(target)

    if not inputs:
        # return empty tensors on the correct device
        return (
            torch.zeros(0, 1, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        )

    x_inputs = pad_sequence(inputs, batch_first=True, padding_value=0).to(device)
    y_targets = torch.tensor(targets, dtype=torch.long, device=device)
    return x_inputs, y_targets

def train_rl(  # noqa: PLR0913, PLR0915, C901
    model: QNetwork,
    train_sequences: torch.Tensor,
    val_sequences: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    epochs: int = 10,
    patience: int = 3,
    *,
    window_size: int | None = None,
    gamma: float = 0.99  # Discount factor for future rewards
) -> QNetwork:
    """
    Train QNetwork to converge to N-gram statistics using cross-entropy loss.

    This implementation uses Maximum Likelihood Estimation (MLE) through cross-entropy,
    which should converge to the same empirical distribution as N-gram frequency counts.

    Key features for proper convergence:
    - Sufficient model capacity (embedding_dim and hidden_dim should be large enough)
    - Proper learning rate (use adaptive optimizer like Adam)
    - See all training examples multiple times (epochs)
    - Early stopping based on validation accuracy

    If window_size is provided, each sampled prefix is cropped to its last `window_size` tokens.
    """
    device = model.device or train_sequences.device
    _ = criterion  # Unused, we use cross-entropy
    _ = gamma  # Not used in this supervised learning approach
    model = model.to(device)

    # Build complete dataset of all prefixes and their targets
    # This ensures we train on the full empirical distribution
    prefixes: list[torch.Tensor] = []
    targets: list[int] = []

    for i in range(train_sequences.shape[0]):
        seq = train_sequences[i]
        valid_len = int((seq != 0).sum().item())
        if valid_len < RL_MIN_PREFIX_LEN:
            continue

        # Create all possible prefixes (like N-gram does)
        for k in range(1, valid_len):
            prefix = seq[:k].clone()
            if window_size is not None and prefix.numel() > window_size:
                prefix = prefix[-window_size:]
            prefixes.append(prefix)
            targets.append(int(seq[k].item()))

    if not prefixes:
        return model

    # Create dataset and dataloader
    class PrefixDataset(torch.utils.data.Dataset):
        def __init__(self, xs: list[torch.Tensor], ys: list[int]) -> None:
            self.xs = xs
            self.ys = ys

        def __len__(self) -> int:
            return len(self.xs)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            return self.xs[idx], self.ys[idx]

    def collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        xs = [b[0] for b in batch]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        x_pad = pad_sequence(xs, batch_first=True, padding_value=0)
        return x_pad, ys

    dataset = PrefixDataset(prefixes, targets)
    # Shuffle to avoid order bias and help convergence
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
    )

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    logger.info("Training on %d prefix examples to learn empirical distribution", len(prefixes))

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        with tqdm(total=len(loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for x_pad, y_batch_loop in loader:
                x_batch = x_pad.to(device)
                y_batch = y_batch_loop.to(device)

                # Forward pass: get Q-values (logits) for all actions
                outputs = model(x_batch)
                # Some model.forward() implementations return (logits, hidden). We only need logits.
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                # outputs: [batch, seq_len, vocab]
                # We need logits for the last (non-padding) timestep of each prefix in the batch.
                lengths = (x_batch != 0).sum(dim=1)
                # last index per row (lengths >= 1 since prefixes are non-empty)
                last_idx = lengths - 1
                batch_idx = torch.arange(x_batch.size(0), device=outputs.device)
                q_values = outputs[batch_idx, last_idx, :]  # [batch_size, vocab_size]

                # Cross-entropy loss = negative log-likelihood = MLE training
                # This makes the model learn P(next_activity | prefix) from the data
                loss = F.cross_entropy(q_values, y_batch)

                optimizer.zero_grad()
                loss.backward()

                # Optional: gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Track accuracy
                predicted_actions = torch.argmax(q_values, dim=-1)
                epoch_correct += (predicted_actions == y_batch).sum().item()
                epoch_total += y_batch.size(0)

                epoch_loss += loss.item()
                pbar.update(1)

        avg_loss = epoch_loss / len(loader)
        train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0

        # Validation
        model.eval()
        with torch.no_grad():
            _, _, val_acc, _ = evaluate_rl(model, val_sequences, window_size=window_size)

        logger.info(
            "Epoch %d: Loss=%.4f, Train Acc=%.4f, Val Acc=%.4f",
            epoch,
            avg_loss,
            train_acc,
            val_acc,
        )

        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            logger.info("New best validation accuracy: %.4f", val_acc)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping! Best val accuracy: %.4f", best_val_acc)
                break

    # Restore best model (the one that generalizes best)
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_rl(  # noqa: C901, PLR0912
    model: QNetwork,
    sequences: torch.Tensor,
    *,
    max_k: int = 3,
    idx_to_activity: dict[int, ActivityName] | None = None,
    window_size: int | None = None,
) -> tuple[dict[str, float | list[int]], list[float], float, list[str]]:
    """

    Evaluate QNetwork in batch mode by predicting next token for every prefix in each sequence.

    Returns (stats, perplexities, eval_time, predicted_vector). Perplexities is empty for RL.
    predicted_vector is a flattened list of predicted next-activity names/indices.

    """
    eval_start = time.time()
    pause_time = 0.0

    model.eval()
    # Determine the device the model is on. Prefer explicit model.device when provided;
    # otherwise fall back to the device of model parameters.
    model_device = getattr(model, "device", None)
    if model_device is None:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = None
    correct = 0
    total = 0
    top_k_correct = [0] * max_k
    predicted_vector: list[str] = []

    with torch.no_grad():
        for i in range(sequences.shape[0]):
            seq = sequences[i]
            # build list of non-padding positions
            valid_len = int((seq != 0).sum().item())
            if valid_len < RL_MIN_PREFIX_LEN:
                continue
            # iterate over prefixes
            for k in range(1, valid_len):
                prefix = seq[:k].unsqueeze(0)  # [1, k]
                if window_size is not None and prefix.shape[1] > window_size:
                    prefix = prefix[:, -window_size:]
                # Ensure prefix is on the same device as the model to avoid
                # "Expected all tensors to be on the same device" runtime error.
                if model_device is not None:
                    prefix = prefix.to(device=model_device)
                outputs = model(prefix)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # outputs may have shape [1, seq_len, vocab] or [1, 1, vocab]
                # or (in some forward implementations) [seq_len, vocab].
                # We need the logits for the last timestep of the prefix.
                if outputs.dim() == 3:
                    # [batch=1, seq_len, vocab] -> take last timestep
                    q_vals = outputs[0, -1, :]
                elif outputs.dim() == 2:
                    # Could be [1, vocab] or [seq_len, vocab]; in both cases take last row
                    q_vals = outputs[-1, :]
                elif outputs.dim() == 1:
                    # Already [vocab]
                    q_vals = outputs
                else:
                    # Fallback: flatten and take last vocab-sized chunk if possible
                    q_flat = outputs.view(-1)
                    if q_flat.numel() >= 1:
                        q_vals = q_flat[-1:]
                    else:
                        raise RuntimeError("Unexpected logits shape in evaluate_rl")

                # Ensure k does not exceed vocabulary size
                k_eff = min(max_k, q_vals.numel())
                topk = torch.topk(q_vals, k=k_eff)
                # topk.indices is 1-D (length k_eff); extract the top-1 prediction safely
                pred_idx = int(topk.indices[0].item())

                # map to activity name if provided
                if idx_to_activity is not None and pred_idx in idx_to_activity:
                    predicted_vector.append(str(idx_to_activity[pred_idx]))
                else:
                    predicted_vector.append(str(pred_idx))

                target_idx = int(seq[k].item())
                total += 1
                if pred_idx == target_idx:
                    correct += 1
                    for j in range(max_k):
                        top_k_correct[j] += 1
                else:
                    # count inclusion in top-k
                    for j in range(max_k):
                        if target_idx in {int(x) for x in topk.indices[: j + 1].tolist()}:
                            top_k_correct[j] += 1

    accuracy = (correct / total) if total > 0 else 0.0
    stats = {
        "accuracy": accuracy,
        "total_predictions": total,
        "correct_predictions": correct,
        "top_k_correct_preds": top_k_correct,
    }
    eval_time = time.time() - eval_start - pause_time
    perplexities: list[float] = []  # not computed for RL
    return stats, perplexities, eval_time, predicted_vector


def train_rnn(  # noqa: C901, PLR0912, PLR0913, PLR0915
    model: LSTMModel | TransformerModel,
    train_sequences: torch.Tensor,
    val_sequences: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    epochs: int = 10,
    patience: int = 3,
    *,
    window_size: int | None = None,
) -> LSTMModel | TransformerModel:
    """
    Train the RNN model on the training set and evaluate on the validation set.

    Returns the trained model.
    """
    dataset = torch.utils.data.TensorDataset(train_sequences)  # Create dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Create dataloader

    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0
    # Determine model device robustly: prefer explicit model.device, else fall back to param device
    model_device = getattr(model, "device", None)
    if model_device is None:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Create progress bar for the training loop
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for batch in dataloader:
                sequences = batch[0]

                # Build teacher-forcing pairs: x is all but last token, y is shifted by one
                x_full = sequences[:, :-1]
                y_full = sequences[:, 1:]

                # If a window size is provided, slice the last `window_size` NON-PADDING tokens per row.
                if window_size is not None:
                    x_slices: list[torch.Tensor] = []
                    y_slices: list[torch.Tensor] = []
                    # Compute per-row effective lengths (exclude padding=0)
                    lengths = (x_full != 0).sum(dim=1).tolist()
                    for b, eff_len in enumerate(lengths):
                        eff = int(eff_len)
                        if eff <= 0:
                            # Skip empty rows entirely (they contribute nothing to loss)
                            continue
                        start = max(0, eff - window_size)
                        x_slices.append(x_full[b, start:eff].clone())
                        y_slices.append(y_full[b, start:eff].clone())

                    # If no valid slices (all rows were empty), skip this batch
                    if len(x_slices) == 0:
                        pbar.update(1)
                        continue

                    # Re-pad windows to a batch tensor
                    x_batch = pad_sequence(x_slices, batch_first=True, padding_value=0)
                    y_batch = pad_sequence(y_slices, batch_first=True, padding_value=0)
                else:
                    x_batch = x_full
                    y_batch = y_full

                # Move tensors to the model device when available to avoid device mismatch
                if model_device is not None:
                    x_batch = x_batch.to(model_device)
                    y_batch = y_batch.to(model_device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(x_batch)
                # Some model forward() implementations (e.g., LSTMModel/GRUModel/QNetwork) return
                # a tuple (logits, hidden). We only need the logits for the loss below.
                if isinstance(outputs, tuple):  # (logits, hidden)
                    outputs = outputs[0]

                # Loss computation:
                # Token-level loss on all non-padding positions
                logits = outputs.view(-1, outputs.shape[-1])  # [batch * seq_len, vocab]
                targets = y_batch.reshape(-1)  # [batch * seq_len]

                mask = targets != 0
                logits_masked = logits[mask]
                targets_masked = targets[mask]

                if logits_masked.size(0) > 0:
                    loss = criterion(logits_masked, targets_masked)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                pbar.update(1)

        msg = f"Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss / len(dataloader):.4f}"
        logger.info(msg)

        # Evaluate on validation set after each epoch
        msg = "Evaluating on validation set..."
        logger.info(msg)
        # updated unpacking to accept extra returned predicted-vector (ignored here)
        stats, _, _, _ = evaluate_rnn(model, val_sequences, window_size=window_size)
        val_accuracy = stats["accuracy"]

        if not isinstance(val_accuracy, float):
            msg = "Validation accuracy is not a float. Check the evaluation function."
            logger.error(msg)
            raise TypeError(msg)

        # Check if current validation accuracy is better than the best recorded accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())  # Save the model state
            patience_counter = 0  # Reset patience counter
            msg = f"New best validation accuracy: {val_accuracy * 100:.2f}%"
            logger.info(msg)
        else:
            patience_counter += 1
            msg = f"Validation accuracy did not improve. Patience counter: {patience_counter}/{patience}"
            logger.info(msg)

        # Early stopping: Stop training if no improvement after patience epochs
        if patience_counter >= patience:
            msg = "Early stopping triggered. Restoring best model weights."
            logger.info(msg)

            msg = f"Best validation accuracy: {best_val_accuracy * 100:.2f}%"
            logger.info(msg)

            if best_model_state:
                model.load_state_dict(best_model_state)
            break

    # Load the best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


def evaluate_rnn(  # noqa: C901, PLR0913, PLR0915
    model: LSTMModel | TransformerModel,
    sequences: torch.Tensor,
    *,
    per_sequence_perplexity: bool = True,
    max_k: int = 3,
    idx_to_activity: dict[int, ActivityName] | None = None,
    window_size: int | None = None,
) -> tuple[dict[str, float | list[int]], list[float], float, list[str]]:
    """
    Evaluate the LSTM/Transformer model on a dataset (train, test, or validation).

    Returns:
        - stats (dict): accuracy and other counts
        - perplexities (list[float]): per-sequence (or aggregated) perplexities
        - eval_time (float): evaluation runtime (seconds)
        - predicted_vector (list[str]): flattened vector of predicted next-activity names
          (one entry per non-padding prediction). If idx_to_activity is not provided,
          indices are stringified.

    """
    eval_start_time = time.time()
    pause_time = 0.0

    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    # Determine device robustly: prefer explicit model.device, else fall back to parameter device
    model_device = getattr(model, "device", None)
    if model_device is None:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = None

    # Initialize list to count top-k correct predictions
    top_k_correct_preds = [0] * max_k

    total_nll = 0.0  # Accumulate negative log-likelihood
    token_count = 0
    perplexities = []

    # New: collect predicted activity names (as strings) for each non-padding prediction
    predicted_vector: list[str] = []

    with torch.no_grad():
        # Default evaluation for LSTM/Transformer (no compaction)
        for i in range(sequences.size(0)):  # Iterate through sequences by index
            single_sequence_trace = sequences[i]

            # Input is all but the last token, target is the sequence shifted by one
            x_input_cpu = single_sequence_trace[:-1]
            y_target_cpu = single_sequence_trace[1:]

            # If windowing is requested, keep only the last `window_size` NON-PADDING tokens.
            if window_size is not None:
                # effective length (exclude padding=0) for x_input_cpu
                eff_len = int((x_input_cpu != 0).sum().item())
                if eff_len > 0:
                    start = max(0, eff_len - window_size)
                    x_input_cpu = x_input_cpu[start:eff_len]
                    y_target_cpu = y_target_cpu[start:eff_len]

            x_input = x_input_cpu.unsqueeze(0)
            y_target = y_target_cpu.unsqueeze(0)

            if model_device is not None:
                x_input = x_input.to(device=model_device)
                y_target = y_target.to(device=model_device)

            outputs = model(x_input)
            if isinstance(outputs, tuple):  # handle (logits, hidden)
                outputs = outputs[0]

            # Flatten for comparison
            predicted_indices = torch.argmax(outputs, dim=-1)
            predicted_indices = predicted_indices.view(-1)
            y_target = y_target.view(-1)

            # Create a mask to ignore padding
            mask = y_target != 0  # Mask for non-padding targets
            masked_targets = y_target[mask]

            # Save predicted values for non-padding positions, mapped to activity names
            if mask.sum().item() > 0:
                masked_predicted = predicted_indices[mask].cpu().tolist()
                if idx_to_activity is not None:
                    mapped = [idx_to_activity.get(int(idx), str(int(idx))) for idx in masked_predicted]
                else:
                    # Fallback: stringify indices so callers can still inspect values
                    mapped = [str(int(idx)) for idx in masked_predicted]
                predicted_vector.extend(mapped)  # type: ignore # noqa: PGH003

            # Apply the mask and count correct predictions
            correct_predictions += (predicted_indices[mask] == masked_targets).sum().item()
            total_predictions += mask.sum().item()  # Count non-padding tokens

            pause_start_time = time.time()

            # ================ Start for metrics ================

            # Get top-k predictions for each position
            # Shape: [batch_size, seq_len, k]
            _, top_k_indices = torch.topk(outputs, k=max_k, dim=-1)

            # Reshape top_k_indices to [batch_size*seq_len, k]
            top_k_indices = top_k_indices.view(-1, max_k)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)  # Log probabilities
            log_probs = log_probs.view(-1, log_probs.shape[-1])
            masked_log_probs = log_probs[mask]

            # Apply mask to top-k predictions
            masked_top_k = top_k_indices[mask]

            # Count correct predictions for each k
            for k in range(max_k):
                # For each position, check if the true label is in the top k predictions
                # Get the predictions up to k+1 (inclusive) for each position
                top_k_preds = masked_top_k[:, : (k + 1)]

                # Check if true label is in the top-k predictions for each position
                # Expand target to match shape of predictions for comparison
                expanded_targets = masked_targets.unsqueeze(1).expand_as(top_k_preds)

                # Count positions where the true label appears in the top-k predictions
                top_k_correct = (top_k_preds == expanded_targets).any(dim=1).sum().item()
                top_k_correct_preds[k] += int(top_k_correct)

            # NLL for perplexity
            token_log_probs = masked_log_probs[torch.arange(len(masked_targets)), masked_targets]

            # Non-classical way (to match ngrams)
            sequence_nll = -token_log_probs.sum().item()  # Negative log likelihood
            sequence_length = masked_targets.size(0)

            if per_sequence_perplexity:
                # Calculate perplexity for the sequence
                sequence_perplexity = (
                    torch.exp(torch.tensor(sequence_nll / sequence_length)).item()
                    if sequence_length > 0
                    else float("inf")
                )
                perplexities.append(sequence_perplexity)

            total_nll += sequence_nll
            token_count += sequence_length

            # ================= End for metrics =================
            pause_time += time.time() - pause_start_time

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    pause_start_time = time.time()
    if not per_sequence_perplexity:
        perplexities.append(
            torch.exp(torch.tensor(total_nll / token_count)).item() if token_count > 0 else float("inf")
        )

    logger.debug("Perplexity: %s", perplexities[-1] if len(perplexities) > 0 else None)

    stats = {
        "accuracy": accuracy,
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions,
        "top_k_correct_preds": top_k_correct_preds,
    }
    pause_time += time.time() - pause_start_time
    eval_time = time.time() - eval_start_time - pause_time

    return stats, perplexities, eval_time, predicted_vector
