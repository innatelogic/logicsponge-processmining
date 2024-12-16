import copy
import logging

import torch
import torch.utils.data
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


class LSTMModel(nn.Module):
    device: torch.device | None
    embedding: nn.Embedding
    lstm1: nn.LSTM
    lstm2: nn.LSTM
    fc: nn.Linear

    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, device: torch.device | None = None
    ):
        super().__init__()
        self.device = device
        # Use padding_idx=0 to handle padding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0, device=device)

        # Two LSTM layers
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, device=device)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, device=device)
        # self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim, device=device)

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # Convert activity indices to embeddings

        # Pass through LSTM layers
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)
        # lstm_out, _ = self.lstm3(lstm_out)

        return self.fc(lstm_out)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization for linear layers
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize biases to zero
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)  # Xavier initialization for input-hidden weights
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)  # Orthogonal initialization for hidden-hidden weights
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)  # Initialize biases to zero
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.1, 0.1)  # Uniform initialization for embedding weights


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

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Create progress bar for the training loop
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for batch in dataloader:
                sequences = batch[0]

                # Input is the entire sequence except the last element (predict all positions)
                x_batch = sequences[:, :-1]  # All but the last token are input
                y_batch = sequences[:, 1:]  # Target: sequence shifted by one

                optimizer.zero_grad()

                # Forward pass
                outputs = model(x_batch)

                # Reshape the outputs and targets for loss computation
                outputs = outputs.view(-1, outputs.shape[-1])  # [batch_size * sequence_length, output_dim]
                y_batch = y_batch.reshape(-1)  # Flatten the target for CrossEntropyLoss

                # Create a mask for positions that are not padding (non-zero indices)
                mask = y_batch != 0  # Mask for non-padding targets

                # Apply the mask to outputs and targets
                outputs = outputs[mask]
                y_batch = y_batch[mask]

                # Compute the loss only for non-padding positions
                loss = criterion(outputs, y_batch)

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
        val_accuracy = evaluate_rnn(model, val_sequences)

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


def evaluate_rnn(model, sequences, dataset_type="Validation"):
    """
    Evaluate the LSTM model on a dataset (train, test, or validation).
    Returns accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for sequence in sequences:
            # Input is all but the last token, target is the sequence shifted by one
            x_input = sequence[:-1].unsqueeze(0)  # All but the last token
            y_target = sequence[1:].unsqueeze(0)  # Shifted by one as target

            outputs = model(x_input)
            predicted_indices = torch.argmax(outputs, dim=-1)

            # Flatten for comparison
            predicted_indices = predicted_indices.view(-1)
            y_target = y_target.view(-1)

            # Create a mask to ignore padding
            mask = y_target != 0  # Mask for non-padding targets

            # Apply the mask and count correct predictions
            correct_predictions += (predicted_indices[mask] == y_target[mask]).sum().item()
            total_predictions += mask.sum().item()  # Count non-padding tokens

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    msg = f"{dataset_type} accuracy: {accuracy * 100:.4f}%"
    logger.info(msg)

    return accuracy
