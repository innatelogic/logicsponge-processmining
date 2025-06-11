import copy
import logging
import math
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from logicsponge.processmining.types import Event

logger = logging.getLogger(__name__)


# ============================================================
# Models (RNN and LSTM)
# ============================================================


class RNNModel(nn.Module):
    device: torch.device | None
    embedding: nn.Embedding
    rnn1: nn.RNN
    rnn2: nn.RNN
    fc: nn.Linear

    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, device: torch.device | None = None
    ):
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
        # Convert activity indices to embeddings
        x = self.embedding(x)

        # Pass through layers
        rnn_out, _ = self.rnn1(x)
        rnn_out, _ = self.rnn2(rnn_out)

        return self.fc(rnn_out)


# class LSTMModel(nn.Module):
#     device: torch.device | None
#     embedding: nn.Embedding
#     lstm1: nn.LSTM
#     lstm2: nn.LSTM
#     fc: nn.Linear
#
#     def __init__(
#         self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, device: torch.device | None = None
#     ):
#         super().__init__()
#         self.device = device
#         # Use padding_idx=0 to handle padding
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0, device=device)
#
#         # Two LSTM layers
#         self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, device=device)
#         self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, device=device)
#         # self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
#
#         self.fc = nn.Linear(hidden_dim, output_dim, device=device)
#
#         # Apply custom weight initialization
#         self.apply(self._init_weights)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.embedding(x)  # Convert activity indices to embeddings
#
#         # Pass through LSTM layers
#         lstm_out, _ = self.lstm1(x)
#         lstm_out, _ = self.lstm2(lstm_out)
#         # lstm_out, _ = self.lstm3(lstm_out)
#
#         return self.fc(lstm_out)
#
#     def _init_weights(self, m: nn.Module) -> None:
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)  # Xavier initialization for linear layers
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)  # Initialize biases to zero
#         elif isinstance(m, nn.LSTM):
#             for name, param in m.named_parameters():
#                 if "weight_ih" in name:
#                     nn.init.xavier_uniform_(param.data)  # Xavier initialization for input-hidden weights
#                 elif "weight_hh" in name:
#                     nn.init.orthogonal_(param.data)  # Orthogonal initialization for hidden-hidden weights
#                 elif "bias" in name:
#                     nn.init.constant_(param.data, 0)  # Initialize biases to zero
#         elif isinstance(m, nn.Embedding):
#             nn.init.uniform_(m.weight, -0.1, 0.1)  # Uniform initialization for embedding weights


class LSTMModel(nn.Module):
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
        output_dim: int,
        use_one_hot: bool = False,
        device: torch.device | None = None,
    ):
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
        if not self.use_one_hot and self.embedding is not None:
            # Use embedding layer
            x = self.embedding(x)
        else:
            # Use one-hot encoding
            # print(f"x shape: {x.shape}, dtype: {x.dtype}, unique values: {torch.unique(x)}")
            x = F.one_hot(x, num_classes=self.vocab_size).float().to(self.device)

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


# ============================================================
# Training and Evaluation
# ============================================================


class PreprocessData:
    def __init__(self):
        self.activity_to_idx = {}
        self.idx_to_activity = {}
        self.current_idx = 1  # 0 is reserved for padding

    # Function to get the activity index (for the embedding layer)
    def get_activity_index(self, activity):
        if activity not in self.activity_to_idx:
            self.activity_to_idx[activity] = self.current_idx
            self.idx_to_activity[self.current_idx] = activity
            self.current_idx += 1
        return self.activity_to_idx[activity]

    def preprocess_data(self, dataset: list[list[Event]]):
        processed_sequences = []

        for sequence in dataset:
            index_sequence = [self.get_activity_index(event["activity"]) for event in sequence]  # Convert to indices
            processed_sequences.append(torch.tensor(index_sequence, dtype=torch.long))

        # Pad sequences, using 0 as the padding value
        return pad_sequence(processed_sequences, batch_first=True, padding_value=0)


def train_rnn(model, train_sequences, val_sequences, criterion, optimizer, batch_size, epochs=10, patience=3):
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
        stats, _, _ = evaluate_rnn(model, val_sequences)
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

            model.load_state_dict(best_model_state)
            break

    # Load the best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


def evaluate_rnn(
    model,
    sequences: torch.Tensor,
    dataset_type: str = "Validation",
    *,
    per_sequence_perplexity: bool = True,
    max_k: int = 3,
) -> tuple[dict[str, float | list[int]], list[float], float]:
    """
    Evaluate the LSTM model on a dataset (train, test, or validation).

    Returns accuracy.
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

    with torch.no_grad():
        for i in range(sequences.size(0)):  # Iterate through sequences by index
            single_sequence_trace = sequences[i]

            # Input is all but the last token, target is the sequence shifted by one
            x_input_cpu = single_sequence_trace[:-1]
            y_target_cpu = single_sequence_trace[1:]

            x_input = x_input_cpu.unsqueeze(0)
            y_target = y_target_cpu.unsqueeze(0)

            if model_device is not None:
                x_input = x_input.to(model_device)
                y_target = y_target.to(model_device)

            outputs = model(x_input)

            # Flatten for comparison
            predicted_indices = torch.argmax(outputs, dim=-1)
            predicted_indices = predicted_indices.view(-1)
            y_target = y_target.view(-1)

            # Create a mask to ignore padding
            mask = y_target != 0  # Mask for non-padding targets
            masked_targets = y_target[mask]

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
            log_probs = F.log_softmax(outputs, dim=-1)  # Log probabilities
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

    logger.debug("Perplexity: %s", perplexities[-1])

    stats = {
        "accuracy": accuracy,
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions,
        "top_k_correct_preds": top_k_correct_preds,
    }

    pause_time += time.time() - pause_start_time

    eval_time = time.time() - eval_start_time - pause_time

    return stats, perplexities, eval_time


class SimpleTransformerModel(nn.Module):
    device: torch.device | None
    embedding: nn.Embedding | None
    use_one_hot: bool
    vocab_size: int
    embedding_dim: int
    pos_embedding: nn.Parameter
    transformer: nn.TransformerEncoder
    fc: nn.Linear

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_one_hot: bool = False,
        device: torch.device | None = None,
        max_seq_len: int = 512,  # You can adjust this if needed
    ):
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
        if not self.use_one_hot and self.embedding is not None:
            x = self.embedding(x)
        else:
            x = F.one_hot(x, num_classes=self.vocab_size).float().to(self.device)

        # Add positional encoding (crop to match sequence length)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]

        # Pass through transformer
        x = self.transformer(x)

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


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 5000, device: torch.device | None = None):
        super().__init__()
        self.device = device

        pe = torch.zeros(max_len, embedding_dim, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]  # type: ignore


class TransformerModel(nn.Module):
    device: torch.device | None
    embedding: nn.Embedding | None
    use_one_hot: bool
    vocab_size: int
    embedding_dim: int
    pos_encoding: PositionalEncoding
    transformer: nn.Transformer
    fc: nn.Linear
    input_projection: nn.Linear | None  # Added for one-hot projection

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_one_hot: bool = False,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device
        self.use_one_hot = use_one_hot
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Conditional embedding layer or input projection for one-hot
        self.input_projection = None
        if not use_one_hot:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0, device=device)
        else:
            self.embedding = None  # Ensure embedding is None if using one-hot
            self.input_projection = nn.Linear(vocab_size, embedding_dim, device=device)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, device=device)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            device=device,
        )

        # Output layer
        self.fc = nn.Linear(embedding_dim, vocab_size, device=device)

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:  # Renamed x to token_indices
        # token_indices shape: (batch, seq_len)
        current_device = self.device if self.device is not None else token_indices.device

        # Create padding mask from token_indices.
        # True for positions that are padded (original token_id == 0).
        padding_mask = (token_indices == 0).to(current_device)

        x_embedded: torch.Tensor
        if not self.use_one_hot:
            if self.embedding is None:
                raise ValueError("Embedding layer is None, but use_one_hot is False.")
            x_embedded = self.embedding(token_indices) * math.sqrt(self.embedding_dim)
        else:
            if self.input_projection is None:
                raise ValueError("Input projection layer is None, but use_one_hot is True.")
            x_one_hot = F.one_hot(token_indices, num_classes=self.vocab_size).float().to(current_device)
            x_embedded = self.input_projection(x_one_hot) * math.sqrt(self.embedding_dim)

        # Add positional encoding
        x_processed = self.pos_encoding(x_embedded)  # Shape: (batch, seq_len, embedding_dim)

        # Create causal mask for the decoder's self-attention.
        seq_len = x_processed.size(1)
        causal_mask = self.transformer.generate_square_subsequent_mask(seq_len).to(current_device)

        # Pass through the nn.Transformer module.
        # src is x_processed, tgt is also x_processed (teacher forcing for next-token prediction).
        # src_key_padding_mask and tgt_key_padding_mask are derived from original padding.
        # memory_key_padding_mask also uses padding_mask as memory comes from encoder based on src.
        transformer_output = self.transformer(
            src=x_processed,
            tgt=x_processed,
            tgt_mask=causal_mask,
            src_key_padding_mask=padding_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask,
        )
        # transformer_output shape: (batch, seq_len, embedding_dim)

        return self.fc(transformer_output)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):  # Covers self.fc and self.input_projection
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding) and m is not None:
            nn.init.uniform_(m.weight, -0.1, 0.1)


def train_transformer(model, train_sequences, val_sequences, criterion, optimizer, batch_size, epochs=10, patience=3):
    """Training function specifically for Transformer model."""
    dataset = torch.utils.data.TensorDataset(train_sequences)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0
    model_device = model.device  # Get model's device once

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for batch in dataloader:
                sequences = batch[0]
                model_device = model.device  # Get model's device

                # Input is the entire sequence except the last element
                x_batch = sequences[:, :-1]
                y_batch = sequences[:, 1:]

                if model_device is not None:
                    x_batch = x_batch.to(model_device)
                    y_batch = y_batch.to(model_device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(x_batch)

                # Reshape for loss computation
                outputs = outputs.view(-1, outputs.shape[-1])
                y_batch = y_batch.reshape(-1)

                # Create mask for non-padding positions
                mask = y_batch != 0

                # Apply mask
                outputs_masked = outputs[mask]
                y_batch_masked = y_batch[mask]

                # Compute loss
                if outputs_masked.size(0) > 0:  # Ensure there are non-padded targets
                    loss = criterion(outputs_masked, y_batch_masked)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                pbar.update(1)

        msg = f"Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss / len(dataloader):.4f}"
        logger.info(msg)

        # Evaluate on validation set
        msg = "Evaluating on validation set..."
        logger.info(msg)
        stats, _, _ = evaluate_transformer(model, val_sequences)
        val_accuracy = stats["accuracy"]

        if not isinstance(val_accuracy, float):
            msg = "Validation accuracy is not a float. Check the evaluation function."
            logger.error(msg)
            raise TypeError(msg)

        # Check for improvement
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            msg = f"New best validation accuracy: {val_accuracy * 100:.2f}%"
            logger.info(msg)
        else:
            patience_counter += 1
            msg = f"Validation accuracy did not improve. Patience counter: {patience_counter}/{patience}"
            logger.info(msg)

        # Early stopping
        if patience_counter >= patience:
            msg = "Early stopping triggered. Restoring best model weights."
            logger.info(msg)
            msg = f"Best validation accuracy: {best_val_accuracy * 100:.2f}%"
            logger.info(msg)
            model.load_state_dict(best_model_state)
            break

    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


def evaluate_transformer(
    model,
    sequences: torch.Tensor,
    dataset_type: str = "Validation",
    *,
    per_sequence_perplexity: bool = True,
    max_k: int = 3,
) -> tuple[dict[str, float | list[int]], list[float], float]:
    """Evaluate the Transformer model on a dataset."""
    eval_start_time = time.time()
    pause_time = 0.0

    model.eval()
    correct_predictions = 0
    total_predictions = 0
    model_device = model.device  # Get model's device once

    # Initialize list to count top-k correct predictions
    top_k_correct_preds = [0] * max_k

    total_nll = 0.0
    token_count = 0
    perplexities = []

    with torch.no_grad():
        for i in range(sequences.size(0)):  # Iterate through sequences by index
            single_sequence_trace = sequences[i]

            # Input is all but the last token
            x_input_cpu = single_sequence_trace[:-1]
            y_target_cpu = single_sequence_trace[1:]

            x_input = x_input_cpu.unsqueeze(0)
            y_target = y_target_cpu.unsqueeze(0)

            if model_device is not None:
                x_input = x_input.to(model_device)
                y_target = y_target.to(model_device)

            outputs = model(x_input)

            # Flatten for comparison
            predicted_indices = torch.argmax(outputs, dim=-1)
            predicted_indices = predicted_indices.view(-1)
            y_target = y_target.view(-1)

            # Create a mask to ignore padding
            mask = y_target != 0
            masked_targets = y_target[mask]

            # Count correct predictions
            correct_predictions += (predicted_indices[mask] == masked_targets).sum().item()
            total_predictions += mask.sum().item()

            pause_start_time = time.time()

            # Top-k predictions
            _, top_k_indices = torch.topk(outputs, k=max_k, dim=-1)
            top_k_indices = top_k_indices.view(-1, max_k)
            log_probs = F.log_softmax(outputs, dim=-1)
            log_probs = log_probs.view(-1, log_probs.shape[-1])
            masked_log_probs = log_probs[mask]

            masked_top_k = top_k_indices[mask]

            # Count correct predictions for each k
            for k in range(max_k):
                top_k_preds = masked_top_k[:, : (k + 1)]
                expanded_targets = masked_targets.unsqueeze(1).expand_as(top_k_preds)
                top_k_correct = (top_k_preds == expanded_targets).any(dim=1).sum().item()
                top_k_correct_preds[k] += int(top_k_correct)

            # NLL for perplexity
            token_log_probs = masked_log_probs[torch.arange(len(masked_targets)), masked_targets]
            sequence_nll = -token_log_probs.sum().item()
            sequence_length = masked_targets.size(0)

            if per_sequence_perplexity:
                sequence_perplexity = (
                    torch.exp(torch.tensor(sequence_nll / sequence_length)).item()
                    if sequence_length > 0
                    else float("inf")
                )
                perplexities.append(sequence_perplexity)

            total_nll += sequence_nll
            token_count += sequence_length

            pause_time += time.time() - pause_start_time

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    pause_start_time = time.time()
    if not per_sequence_perplexity:
        perplexities.append(
            torch.exp(torch.tensor(total_nll / token_count)).item() if token_count > 0 else float("inf")
        )

    logger.debug("Perplexity: %s", perplexities[-1])

    stats = {
        "accuracy": accuracy,
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions,
        "top_k_correct_preds": top_k_correct_preds,
    }

    pause_time += time.time() - pause_start_time
    eval_time = time.time() - eval_start_time - pause_time

    return stats, perplexities, eval_time
