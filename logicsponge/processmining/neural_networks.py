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

from logicsponge.processmining.types import ActivityName, Event

logger = logging.getLogger(__name__)


# ============================================================
# Models (RNN and LSTM)
# ============================================================



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
        if self.device is not None and x.device != self.device:
            x = x.to(self.device)
        if hidden is not None and self.device is not None:
            try:
                if hidden.device != self.device:
                    hidden = hidden.to(self.device)
            except AttributeError:
                # hidden may be a tuple in future variants
                pass
        if not self.use_one_hot and self.embedding is not None:
            emb = self.embedding(x)  # [batch, seq_len, embedding_dim]
        else:
            # Convert indices to one-hot; ensure padding index 0 -> all-zeros vector
            indices = x
            emb = F.one_hot(indices, num_classes=self.vocab_size).float().to(indices.device)
            pad_mask = (indices == 0).unsqueeze(-1)
            emb.masked_fill_(pad_mask, 0.0)

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
        if self.device is not None and input_token.device != self.device:
            input_token = input_token.to(self.device)
        if hidden is not None and self.device is not None:
            try:
                if hidden.device != self.device:
                    hidden = hidden.to(self.device)
            except AttributeError:
                pass
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)

        if not self.use_one_hot and self.embedding is not None:
            emb = self.embedding(input_token)  # [batch, 1, embedding_dim]
        else:
            # Convert indices to one-hot; ensure padding index 0 -> all-zeros vector
            indices = input_token
            emb = F.one_hot(indices, num_classes=self.vocab_size).float().to(indices.device)
            pad_mask = (indices == 0).unsqueeze(-1)
            emb.masked_fill_(pad_mask, 0.0)

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
        # Ensure input on model device
        if self.device is not None and x.device != self.device:
            x = x.to(self.device)
        if hidden is not None and self.device is not None and hidden.device != self.device:
            hidden = hidden.to(self.device)
        if not self.use_one_hot and self.embedding is not None:
            x = self.embedding(x)
        else:
            indices = x
            x = F.one_hot(indices, num_classes=self.vocab_size).float().to(indices.device)
            pad_mask = (indices == 0).unsqueeze(-1)
            x.masked_fill_(pad_mask, 0.0)

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
        # Ensure on model device
        if self.device is not None and input_token.device != self.device:
            input_token = input_token.to(self.device)
        if hidden is not None and self.device is not None and hidden.device != self.device:
            hidden = hidden.to(self.device)
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

    """

    device: torch.device | None
    embedding: nn.Embedding | None
    use_one_hot: bool
    vocab_size: int
    embedding_dim: int
    lstm1: nn.LSTM
    lstm2: nn.LSTM
    fc: nn.Linear

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        *,
        use_one_hot: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the LSTM model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the hidden layers in the LSTM.
            output_dim (int): Dimension of the output layer.
            use_one_hot (bool): Whether to use one-hot encoding instead of embeddings.
            device (torch.device | None): Device to run the model on (CPU or GPU).

        """
        super().__init__()
        self.device = device
        self.use_one_hot = use_one_hot
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Conditional embedding layer
        if not use_one_hot:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0, device=device)
        else:
            self.embedding = None

        # Input dimension to LSTM depends on the encoding method
        input_dim = vocab_size if use_one_hot else embedding_dim

        # LSTM layers
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, device=device)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, device=device)

        self.fc = nn.Linear(hidden_dim, input_dim, device=device)

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len), where each element is an activity index.

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
            x = torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float().to(self.device)

        # Pass through LSTM layers
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)

        return self.fc(lstm_out)

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
    transformer: nn.TransformerEncoder
    fc: nn.Linear

    def __init__( # noqa: PLR0913
        self,
        vocab_size: int,
        *,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        padding_idx: int = 0,
        attention_heads: int = 4,
        use_one_hot: bool = False,
        device: torch.device | None = None,
        pos_encoding_type: str = "rope",
    ) -> None:
        """
        Initialize the Transformer model.

        Args:
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

        # store positional encoding selection
        self.pos_encoding_type = pos_encoding_type

        # Model dimension for the Transformer (d_model) depends on representation
        # - If embeddings are used, d_model = embedding_dim
        # - If one-hot is used, d_model = vocab_size
        d_model = embedding_dim if not use_one_hot else vocab_size
        self.d_model = d_model

        # Conditional embedding layer
        self.embedding = (
            nn.Embedding(vocab_size, embedding_dim, padding_idx=0, device=device) if not use_one_hot else None
        )


        # Transformer encoder
        # Use a custom rotary-aware encoder layer stack so we can apply RoPE to Q/K
        # inside attention. This keeps the rest of the Transformer semantics intact
        # while enabling Rotary Positional Embeddings.
        self.num_heads = attention_heads
        self.num_layers = num_layers
        self._rotary_base = 10000.0
        # Cache for base causal masks per sequence length to avoid reallocation
        self._causal_cache: dict[int, torch.Tensor] = {}
        # AMP control: disabled by default to preserve exact behavior
        self.enable_amp = False  # type: bool
        # Build stack of rotary encoder layers
        self.transformer_layers = nn.ModuleList([
            RotaryTransformerEncoderLayer(d_model, attention_heads, dim_feedforward=hidden_dim, dropout=0.0, device=device)
            for _ in range(num_layers)
        ])

        # Output layer -- produce logits over the vocabulary (vocab_size)
        # Training/evaluation code expects shape [batch, seq_len, vocab_size]
        self.fc = nn.Linear(d_model, vocab_size, device=device)

        # Custom initialization for linear/embedding layers
        self.apply(self._init_weights)



    def _get_base_causal(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Return an upper-triangular causal mask [L,L] (True=masked) for given length.

        Cached on CPU and moved to the requested device on demand.
        """
        cached = self._causal_cache.get(seq_len)
        if cached is None:
            cached = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
            self._causal_cache[seq_len] = cached
        return cached.to(device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: C901, PLR0912, PLR0915
        """
        Forward pass through the Transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len), where each element is an activity index.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim),
            where each element is the predicted activity.

        """
        # Move input to model device if specified, then derive working device from x
        if self.device is not None and x.device != self.device:
            x = x.to(self.device)
        # Ensure we keep computations on the same device as input indices
        x_device = x.device
        # Create padding mask (True for padding positions) from token indices
        key_padding_mask = x == 0

        # Autocast context (disabled by default to avoid behavior change)
        if self.enable_amp:
            amp_device = "cuda" if x_device.type == "cuda" else "cpu"
            amp_dtype = torch.float16 if amp_device == "cuda" else torch.bfloat16
            with torch.autocast(device_type=amp_device, dtype=amp_dtype):
                if not self.use_one_hot and self.embedding is not None:
                    x = self.embedding(x)
                else:
                    indices = x
                    x = F.one_hot(indices, num_classes=self.vocab_size).float().to(x_device)
                    pad_mask = (indices == 0).unsqueeze(-1).to(x_device)
                    x.masked_fill_(pad_mask, 0.0)
        elif not self.use_one_hot and self.embedding is not None:
            x = self.embedding(x)
        else:
            indices = x
            x = F.one_hot(indices, num_classes=self.vocab_size).float().to(x_device)
            pad_mask = (indices == 0).unsqueeze(-1).to(x_device)
            x.masked_fill_(pad_mask, 0.0)

        # Determine batch and sequence dimensions after embedding/one-hot
        batch_size, seq_len, _ = x.size()


                # Compute minimal global left padding across the batch and crop it away
        # to reduce attention compute from O(L^2) to O((L - s)^2) where s is the
        # number of columns that are padding for all rows.
        nonpad = (~key_padding_mask)
        has_any = nonpad.any(dim=1)
        # first non-pad index per row; set to seq_len if no non-pad exists
        first_real = torch.where(
            has_any,
            torch.argmax(nonpad.to(torch.int64), dim=1),
            torch.full((batch_size,), seq_len, device=x_device, dtype=torch.long),
        )  # [B]
        # minimal first_real across batch; columns [0:s) are all padding for everyone
        s = int(first_real.min().item()) if batch_size > 0 else 0
        # Only crop leading columns if there will remain at least one token.
        # Cropping to an empty sequence (seq_len == 0) would cause downstream
        # reductions (e.g. argmax over dim=1) to fail with IndexError.
        if s > 0 and s < seq_len:
            x = x[:, s:, :]
            key_padding_mask = key_padding_mask[:, s:]
            seq_len = x.size(1)
        else:
            # If s == 0 (no common left-pad) or s >= seq_len (all-pad case),
            # avoid cropping. For the all-pad case we'll keep seq_len as-is
            # (which may be 0) and handle empty-dimension logic later.
            s = 0

        # Positional encoding selection.
        # If RoPE is selected, we compute cos/sin and DO NOT add an additive pos embedding.
        cos = sin = None
        # Compute rotary embeddings for head-dim
        head_dim = self.d_model // max(1, getattr(self, "num_heads", 1))
        try:
            cos, sin = _build_rotary_cos_sin(
                seq_len, head_dim, x_device, x.dtype, base=getattr(self, "_rotary_base", 10000.0)
            )
        except Exception:
            cos = sin = None



        base_causal = self._get_base_causal(seq_len, x_device)  # [L,L] True above diagonal
        mask = base_causal.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [B,L,L]

        # Pass through transformer stack (custom rotary-aware layers)
        # Each layer accepts optional cos/sin to apply RoPE; if cos is None, layer runs without RoPE.
        for layer in self.transformer_layers:
            x = layer(
                x,
                src_mask=mask,  # [B,L,L] batch-specific causal mask adapted for left-padding
                src_key_padding_mask=key_padding_mask.to(x_device),
                cos=cos,
                sin=sin,
            )

        # Project to vocabulary logits
        logits_small = self.fc(x)  # [B, L', V]
        # If we cropped leading all-pad columns, left-pad the logits back to original length
        if s > 0:
            V = logits_small.size(-1)
            logits_full = torch.zeros(batch_size, s + seq_len, V, device=logits_small.device, dtype=logits_small.dtype)
            logits_full[:, s:, :] = logits_small
            return logits_full
        return logits_small


    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding) and m is not None:
            nn.init.uniform_(m.weight, -0.1, 0.1)
        # No learned positional parameter; nothing extra to init here




class RotaryTransformerEncoderLayer(nn.Module):
    """
    A simplified Transformer encoder layer that applies RoPE to Q/K inside attention.

    This mirrors the common TransformerEncoderLayer layout but uses explicit
    linear projections for Q/K/V so we can apply RoPE before attention.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.0, device: torch.device | None = None):
        super().__init__()
        if d_model % nhead != 0:
            msg = "d_model must be divisible by nhead"
            raise ValueError(msg)
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, device=device)
        self.k_proj = nn.Linear(d_model, d_model, device=device)
        self.v_proj = nn.Linear(d_model, d_model, device=device)
        self.out_proj = nn.Linear(d_model, d_model, device=device)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device)
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device)

        # Norms and dropout
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.dropout = nn.Dropout(dropout) if dropout is not None and dropout > 0.0 else nn.Identity()
        self.activation = nn.ReLU()

    def forward(
            self, src: torch.Tensor, src_mask: torch.Tensor | None = None,
            src_key_padding_mask: torch.Tensor | None = None,
            cos: torch.Tensor | None = None, sin: torch.Tensor | None = None
        ) -> torch.Tensor:
        """
        Forward pass.

                If cos/sin are provided, apply RoPE to Q/K before attention.
                cos/sin should have shape [seq_len, head_dim//2].

                src_mask semantics (causal/attention blocking mask):
                    - Accepts either shape [L, L] (applied to all batches) or [B, L, L] (per-batch).
                    - Values should be True where attention is DISALLOWED (will be filled with -inf).
                src_key_padding_mask: [B, L] with True at PAD positions (blocks keys/columns).
        """
        # src: [B, seq_len, d_model]
        B, seq_len, _ = src.shape

        # Project
        q = self.q_proj(src)
        k = self.k_proj(src)
        v = self.v_proj(src)

        # reshape to [B, seq_len, nhead, head_dim]
        q = q.view(B, seq_len, self.nhead, self.head_dim)
        k = k.view(B, seq_len, self.nhead, self.head_dim)
        v = v.view(B, seq_len, self.nhead, self.head_dim)

        # Apply rotary if available
        if cos is not None and sin is not None:
            # cos/sin: [seq_len, head_dim//2]
            q = _apply_rotary(q, cos, sin)
            k = _apply_rotary(k, cos, sin)

        # Move heads to [B, nhead, seq_len, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Scaled dot-product attention via SDPA (FlashAttention when available)
        attn_mask = None
        if src_mask is not None:
            if src_mask.dim() == 2:
                attn_mask = src_mask.unsqueeze(0).unsqueeze(0)  # [1,1,L,L]
            elif src_mask.dim() == 3:
                if src_mask.size(0) != B:
                    msg = f"Batch mask first dim {src_mask.size(0)} != batch size {B}"
                    raise ValueError(msg)
                attn_mask = src_mask.unsqueeze(1)  # [B,1,L,L]
            else:
                msg = f"Unsupported src_mask.dim()={src_mask.dim()} (expected 2 or 3)"
                raise ValueError(msg)
        if src_key_padding_mask is not None:
            key_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            attn_mask = key_mask if attn_mask is None else (attn_mask | key_mask)

        # Convert bool mask to additive mask for SDPA if present
        additive_mask = None
        if attn_mask is not None:
            # SDPA expects float additive mask with -inf where disallowed (or a
            # bool in newer versions, but keep compatibility)
            additive_mask = attn_mask.to(q.dtype).masked_fill(attn_mask, float("-inf"))
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=additive_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        if torch.isnan(attn).any():
            attn = torch.nan_to_num(attn, nan=0.0)

        # combine heads
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B, seq_len, self.d_model)
        attn = self.out_proj(attn)

        # Residual + norm
        src2 = self.norm1(src + self.dropout(attn))

        # Feedforward
        ff = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src_out = self.norm2(src2 + self.dropout(ff))
        return src_out


# --------------------------
# Rotary positional helpers
# --------------------------
def _build_rotary_cos_sin(
        seq_len: int, dim: int, device: torch.device, dtype: torch.dtype, base: float = 10000.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build rotary cos and sin matrices for RoPE.

    Returns cos, sin of shape [seq_len, dim//2]. The caller will broadcast
    as needed. `dim` should be the head dimension (must be even).
    """
    if dim % 2 != 0:
        msg = "Head dimension for RoPE must be even"
        raise ValueError(msg)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)  # [seq_len, dim/2]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to tensor x.

    x: [B, seq_len, nhead, head_dim]
    cos/sin: [seq_len, head_dim//2]
    Returns same shape as x with rotation applied to pairs of dimensions.
    """
    # x_even, x_odd
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    # expand cos/sin -> [1, seq_len, 1, head_dim//2]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    # interleave back
    out = torch.zeros_like(x)
    out[..., 0::2] = x_rot_even
    out[..., 1::2] = x_rot_odd
    return out

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


def train_rnn(
    model: LSTMModel | TransformerModel,
    train_sequences: torch.Tensor,
    val_sequences: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    epochs: int = 10,
    patience: int = 3,
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
    model_device = model.device  # Get model's device once

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Create progress bar for the training loop
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for batch in dataloader:
                sequences = batch[0]

                # Input is the entire sequence except the last element (predict all positions)
                x_batch = sequences[:, :-1]  # All but the last token are input
                y_batch = sequences[:, 1:]  # Target: sequence shifted by one

                if window_size is not None:
                    # keep only the last `window_size` timesteps
                    x_batch = x_batch[:, -window_size:]
                    y_batch = y_batch[:, -window_size:]

                if model_device is not None:
                    x_batch = x_batch.to(model_device)
                    y_batch = y_batch.to(model_device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(x_batch)

                # Reshape the outputs and targets for loss computation
                outputs = outputs.view(-1, outputs.shape[-1])  # [batch_size * sequence_length, output_dim]
                y_batch = y_batch.reshape(-1)  # Flatten the target for CrossEntropyLoss

                # Create a mask for positions that are not padding (non-zero indices)
                mask = y_batch != 0  # Mask for non-padding targets

                # Apply the mask to outputs and targets
                outputs_masked = outputs[mask]
                y_batch_masked = y_batch[mask]

                # Compute the loss only for non-padding positions
                if outputs_masked.size(0) > 0:  # Ensure there are non-padded targets
                    loss = criterion(outputs_masked, y_batch_masked)

                    # Backward pass and optimization
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
        stats, _, _, _ = evaluate_rnn(model, val_sequences)
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


def evaluate_rnn(
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
    model_device = model.device  # Get model's device once

    # Initialize list to count top-k correct predictions
    top_k_correct_preds = [0] * max_k

    total_nll = 0.0  # Accumulate negative log-likelihood
    token_count = 0
    perplexities = []

    # New: collect predicted activity names (as strings) for each non-padding prediction
    predicted_vector: list[str] = []

    with torch.no_grad():
        for i in range(sequences.size(0)):  # Iterate through sequences by index
            single_sequence_trace = sequences[i]

            # When window_size is set, we need to make predictions for all positions
            # using a sliding window approach (one prediction per prefix)
            if window_size is not None:
                # Process each prefix position with a sliding window
                full_sequence = single_sequence_trace[single_sequence_trace != 0]  # Remove padding
                seq_len = len(full_sequence)
                
                for pos in range(1, seq_len):  # Start from position 1 (predict after first token)
                    # Get the context window ending at this position
                    start_idx = max(0, pos - window_size)
                    x_input_cpu = full_sequence[start_idx:pos]
                    y_target_cpu = full_sequence[pos:pos+1]
                    
                    x_input = x_input_cpu.unsqueeze(0)
                    y_target = y_target_cpu.unsqueeze(0)
                    
                    if model_device is not None:
                        x_input = x_input.to(device=model_device)
                        y_target = y_target.to(device=model_device)
                    
                    outputs = model(x_input)
                    
                    # Get the prediction for the last position
                    predicted_idx = torch.argmax(outputs[:, -1, :], dim=-1).item()
                    target_idx = y_target.item()
                    
                    # Add to predicted vector
                    if idx_to_activity is not None:
                        pred_str = str(idx_to_activity.get(int(predicted_idx), str(int(predicted_idx))))
                    else:
                        pred_str = str(int(predicted_idx))
                    predicted_vector.append(pred_str)
                    
                    # Update metrics
                    if predicted_idx == target_idx:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    # Top-k and perplexity metrics for this position
                    log_probs = torch.nn.functional.log_softmax(outputs[:, -1, :], dim=-1)
                    _, top_k_indices = torch.topk(outputs[:, -1, :], k=max_k, dim=-1)
                    
                    for k in range(max_k):
                        if target_idx in top_k_indices[0, :(k+1)].tolist():
                            top_k_correct_preds[k] += 1
                    
                    # NLL for perplexity
                    token_log_prob = log_probs[0, int(target_idx)].item()
                    total_nll += -token_log_prob
                    token_count += 1
                
                # Per-sequence perplexity: record after processing all positions in the sequence
                if per_sequence_perplexity and seq_len > 1:
                    # Calculate perplexity for the sequence
                    num_preds_in_seq = seq_len - 1
                    if num_preds_in_seq > 0:
                        # Use only NLL accumulated for this sequence
                        # (we would need to track per-sequence NLL separately for true per-seq perplexity,
                        # but for simplicity we approximate with the overall average)
                        perplexities.append(float("inf"))  # placeholder
            else:
                # Original logic without window_size (process entire sequence at once)
                # Input is all but the last token, target is the sequence shifted by one
                x_input_cpu = single_sequence_trace[:-1]
                y_target_cpu = single_sequence_trace[1:]

                x_input = x_input_cpu.unsqueeze(0)
                y_target = y_target_cpu.unsqueeze(0)

                if model_device is not None:
                    x_input = x_input.to(device=model_device)
                    y_target = y_target.to(device=model_device)

                outputs = model(x_input)

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
                        mapped = [str(idx_to_activity.get(int(idx), str(int(idx)))) for idx in masked_predicted]
                    else:
                        # Fallback: stringify indices so callers can still inspect values
                        mapped = [str(int(idx)) for idx in masked_predicted]
                    predicted_vector.extend(mapped)  # All predictions

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


# Global default: use left-padding for variable-length sequences
LEFT_PAD_DEFAULT = True

def _left_pad_stack(seqs: list[torch.Tensor], *, pad_value: int = 0, target_len: int | None = None) -> torch.Tensor:
    """
    Left-pad 1D LongTensors to a common length and stack as [B, L].

    If target_len is None, pad to the maximum length found in seqs. Assumes all seqs have dtype Long.
    """
    if not seqs:
        return torch.zeros((0, 1), dtype=torch.long)
    max_len = target_len if target_len is not None else max(int(s.numel()) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        l = int(s.numel())
        if l == 0:
            continue
        out[i, max_len - l : max_len] = s[-max_len:]
    return out
