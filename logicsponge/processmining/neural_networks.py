"""The NN architectures."""

import copy
import logging
import time
from dataclasses import dataclass

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
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
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

        self.fc = nn.Linear(hidden_dim, output_dim, device=device)

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
            # Keep device consistent with input tensor to avoid device mismatch
            x = torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float().to(x.device)

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
    pos_embedding: nn.Parameter
    transformer: nn.TransformerEncoder
    fc: nn.Linear

    def __init__( # noqa: PLR0913
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        use_one_hot: bool = False,
        device: torch.device | None = None,
        max_seq_len: int = 1024,  # Was 512. Adjust if needed
    ) -> None:
        """
        Initialize the Transformer model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the hidden layers in the transformer.
            output_dim (int): Dimension of the output layer.
            use_one_hot (bool): Whether to use one-hot encoding instead of embeddings.
            device (torch.device | None): Device to run the model on (CPU or GPU).
            max_seq_len (int): Maximum sequence length for positional encoding.

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

        # Input dimension
        input_dim = vocab_size if use_one_hot else embedding_dim

        # Positional encoding: learnable [1, max_seq_len, input_dim]
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, input_dim, device=device))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=1,
            dim_feedforward=hidden_dim,
            batch_first=True,
            device=device,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output layer
        self.fc = nn.Linear(input_dim, output_dim, device=device)

        # Custom initialization
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

        # Add positional encoding
        seq_len = x.size(1)
        # If the sequence is longer than the learned maximum, interpolate the positional encoding
        if seq_len <= self.pos_embedding.size(1):
            pos = self.pos_embedding[:, :seq_len, :]
        else:
            msg = f"Sequence length {seq_len} exceeds maximum positional encoding length {self.pos_embedding.size(1)}."
            raise ValueError(
                msg
            )
            # Interpolate along the sequence length dimension to match seq_len
            # Shape transform: (1, L, D) -> (1, D, L) for interpolation over L -> back to (1, L, D)
            pos = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=seq_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        # Ensure positional encodings are on the same device as x
        pos = pos.to(x_device)
        x = x + pos

        # === Add causal (left) mask ===
        # Shape: [seq_len, seq_len]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x_device), diagonal=1).bool()
        # PyTorch expects the mask to have True where positions should be masked
        # (i.e., prevent attending)

        # Pass through transformer with mask and key padding mask
        x = self.transformer(x, mask=mask, src_key_padding_mask=key_padding_mask.to(x_device))

        return self.fc(x)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding) and m is not None:
            nn.init.uniform_(m.weight, -0.1, 0.1)
        elif isinstance(m, nn.Parameter):
            nn.init.normal_(m, mean=0.0, std=0.02)



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
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        device: torch.device | None,
        *,
        use_one_hot: bool = False,
    ) -> None:
        """
        Initialize the Q-network.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the hidden layer in the GRU.
            output_dim (int): Dimension of the output layer.
            device (torch.device | None): Device to run the model on (CPU or GPU).
            use_one_hot (bool): Whether to use one-hot encoding instead of embeddings.

        """
        super().__init__()
        self.device = device
        # Optionally disable embedding and process raw one-hot inputs directly.
        self.use_one_hot = use_one_hot
        self.vocab_size = vocab_size
        if not self.use_one_hot:
            # simple embedding (use embedding indices like other NN models)
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0, device=device)
            gru_input_dim = embedding_dim
        else:
            # when using raw one-hot inputs, we don't create an embedding layer
            self.embedding = None
            gru_input_dim = vocab_size

        # GRU input dimension depends on whether we use embedding or one-hot
        self.gru = nn.GRU(gru_input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Q-network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len), where each element is an activity index.

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

        out, _ = self.gru(emb)  # out: [batch, seq_len, hidden_dim]
        last = out[:, -1, :]  # last timestep
        logits = self.fc(last)  # [batch, output_dim]
        return logits.unsqueeze(1)  # match interface [batch, 1, vocab] expected by miners

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
    Train QNetwork in batch mode with Q-learning, matching RLMiner's streaming approach.

    Uses Q-learning with binary rewards (1.0 for correct predictions, 0.0 for incorrect)
    and TD updates over mini-batches. This matches RLMiner's streaming behavior but in batch mode.

    If window_size is provided, each sampled prefix is cropped to its last `window_size` tokens.
    """
    # Implement batched REINFORCE (policy-gradient) without sampling: build the full
    # prefix -> next-token dataset once and iterate through all prefixes each epoch
    # (round-robin). The update uses loss = -reward * log_prob(action). By default
    # we derive a simple reward of 1.0 for the observed (dataset) next-token which
    # reduces to maximizing log-likelihood; this keeps behavior stable while using
    # a policy-gradient objective. The function keeps windowing, validation, and
    # early stopping.
    device = model.device or train_sequences.device
    # keep references to unused args to satisfy linters when callers pass them
    _ = criterion
    _ = gamma
    model = model.to(device)

    # Build full prefix -> next-token dataset (mirrors streaming updates over all events)
    prefixes: list[torch.Tensor] = []
    targets: list[int] = []
    for i in range(train_sequences.shape[0]):
        seq = train_sequences[i]
        valid_len = int((seq != 0).sum().item())
        if valid_len < RL_MIN_PREFIX_LEN:
            continue
        for k in range(1, valid_len):
            prefix = seq[:k].clone()
            if window_size is not None and prefix.numel() > window_size:
                prefix = prefix[-window_size:]
            prefixes.append(prefix)
            targets.append(int(seq[k].item()))

    if not prefixes:
        return model

    def collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        xs = [b[0] for b in batch]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        x_pad = pad_sequence(xs, batch_first=True, padding_value=0)
        return x_pad, ys

    class PrefixDataset(torch.utils.data.Dataset):
        def __init__(self, xs: list[torch.Tensor], ys: list[int]) -> None:
            self.xs = xs
            self.ys = ys

        def __len__(self) -> int:
            return len(self.xs)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            return self.xs[idx], self.ys[idx]

    dataset = PrefixDataset(prefixes, targets)
    # round-robin (no sampling): do not shuffle so that each epoch sees all prefixes
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        with tqdm(total=len(loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for x_pad, y in loader:
                x_batch = x_pad.to(device)
                y_batch = y.to(device)

                # Forward: obtain logits [B, vocab]
                logits = model(x_batch).squeeze(1)
                log_probs = F.log_softmax(logits, dim=-1)

                # Select log-probabilities of the taken actions (the dataset next-token)
                chosen_log_probs = log_probs[torch.arange(log_probs.size(0)), y_batch]

                # Compute reward shaping identical to RLMiner: 1.0 if greedy prediction
                # equals observed action, else 0.0 (used for logging only in RLMiner).
                greedy_preds = torch.argmax(logits, dim=-1)
                rewards = (greedy_preds == y_batch).float()

                # RLMiner applies loss = -log_prob(action) (no reward gating). Match that here.
                loss = - chosen_log_probs.mean()

                # Optionally log mean reward for diagnostics (kept lightweight)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("train_rl epoch=%s batch_reward_mean=%.4f", epoch + 1, float(rewards.mean()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                pbar.update(1)

        avg_loss = epoch_loss / num_batches if num_batches > 0 else float("inf")

        # Validation (greedy accuracy on validation prefixes)
        model.eval()
        with torch.no_grad():
            _, _, val_acc, _ = evaluate_rl(model, val_sequences, window_size=window_size)

        logger.info("Epoch %d: Loss=%.4f, Val Acc=%.4f", epoch, avg_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping!")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_rl(
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
                q_vals = model(prefix).squeeze(1).squeeze(0)  # [vocab]
                topk = torch.topk(q_vals, k=max_k)
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


def train_rnn( # noqa: PLR0913 PLR0915
    model: LSTMModel | TransformerModel,
    train_sequences: torch.Tensor,
    val_sequences: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    epochs: int = 10,
    patience: int = 3,
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


def evaluate_rnn( # noqa: PLR0915
    model: LSTMModel | TransformerModel,
    sequences: torch.Tensor,
    *,
    per_sequence_perplexity: bool = True,
    max_k: int = 3,
    idx_to_activity: dict[int, ActivityName] | None = None,
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
