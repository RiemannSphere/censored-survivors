from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.special import gamma
from pydantic import BaseModel, Field

try:
    from .models import (
        WeibullRNNData,
        WeibullRNNInput,
        WeibullRNNOutput,
        WeibullRNNResult,
        WeibullRNNPrediction,
        WeibullParameters
    )
except ImportError:
    from models import (
        WeibullRNNData,
        WeibullRNNInput,
        WeibullRNNOutput,
        WeibullRNNResult,
        WeibullRNNPrediction,
        WeibullParameters
    )


class WeibullRNNDataset(Dataset):
    """PyTorch Dataset for Weibull RNN data."""
    
    def __init__(self, data: WeibullRNNData):
        """Initialize dataset.
        
        Args:
            data: WeibullRNNData containing sequences and survival data
        """
        self.data = data.data
        
        # Get sequence length and feature names from first record
        first_record = self.data[0]
        self.sequence_length = len(next(iter(first_record.sequence_data.values())))
        self.feature_names = list(first_record.sequence_data.keys())
        self.n_features = len(self.feature_names)
        
        # Convert sequences to tensors
        self.sequences = []
        self.times = []
        self.events = []
        
        for record in self.data:
            # Stack features into a single array
            sequence = np.stack([
                record.sequence_data[name]
                for name in self.feature_names
            ], axis=1)
            
            self.sequences.append(torch.FloatTensor(sequence))
            self.times.append(record.event_time)
            self.events.append(record.event_status)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.sequences[idx],
            torch.FloatTensor([self.times[idx]]),
            torch.FloatTensor([self.events[idx]])
        )


class WeibullRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer with normalization
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # Separate RNNs for shape and scale
        self.shape_rnn = nn.GRU(
            hidden_size, 
            hidden_size, 
            num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.scale_rnn = nn.GRU(
            hidden_size, 
            hidden_size, 
            num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layers with better initialization
        self.shape_out = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.scale_out = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize the final layers with better defaults
        self.shape_out[-1].bias.data.fill_(np.log(2.0))  # Initialize shape close to 2.0
        self.scale_out[-1].bias.data.fill_(np.log(100.0))  # Initialize scale close to 100.0
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed and normalize input
        x = self.embedding(x)
        
        # Process through separate RNNs
        shape_out, _ = self.shape_rnn(x)
        scale_out, _ = self.scale_rnn(x)
        
        # Get last output for each direction
        if self.bidirectional:
            shape_out = torch.cat((shape_out[:, -1, :self.hidden_size],
                                 shape_out[:, 0, self.hidden_size:]), dim=1)
            scale_out = torch.cat((scale_out[:, -1, :self.hidden_size],
                                 scale_out[:, 0, self.hidden_size:]), dim=1)
        else:
            shape_out = shape_out[:, -1]
            scale_out = scale_out[:, -1]
        
        # Output processing with activation functions
        shape = 1.0 + torch.exp(self.shape_out(shape_out))  # Ensure shape > 1
        scale = torch.exp(self.scale_out(scale_out))  # Ensure scale > 0
        
        # Squeeze the outputs to match expected shape
        shape = shape.squeeze(-1)
        scale = scale.squeeze(-1)
        
        return shape, scale


class WeibullLoss(nn.Module):
    """Custom loss function for Weibull Time-To-Event model."""
    
    def forward(
        self,
        shape: torch.Tensor,
        scale: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
    ) -> torch.Tensor:
        """Calculate negative log likelihood loss with improved scale handling.
        
        Args:
            shape: Shape parameter (k)
            scale: Scale parameter (λ)
            time: Observed times
            event: Event indicators
        
        Returns:
            Loss value
        """
        # Ensure positive parameters
        eps = 1e-6
        shape = shape.clamp(min=1.0 + eps)  # Shape should be > 1
        scale = scale.clamp(min=1.0)  # Scale should be at least 1
        time = time.clamp(min=eps)
        
        # Ensure all tensors have the same shape
        shape = shape.view(-1)
        scale = scale.view(-1)
        time = time.view(-1)
        event = event.view(-1)
        
        # Log transform for numerical stability
        log_scale = torch.log(scale)
        log_time = torch.log(time)
        
        # Calculate log likelihood components
        log_likelihood = torch.zeros_like(time)
        
        # For events (uncensored)
        event_mask = event == 1
        if event_mask.any():
            # Use log-space calculations
            log_likelihood[event_mask] = (
                torch.log(shape[event_mask]) - log_scale[event_mask] +
                (shape[event_mask] - 1) * (log_time[event_mask] - log_scale[event_mask]) -
                torch.exp(shape[event_mask] * (log_time[event_mask] - log_scale[event_mask]))
            )
        
        # For censored
        censored_mask = event == 0
        if censored_mask.any():
            log_likelihood[censored_mask] = -(
                torch.exp(shape[censored_mask] * (log_time[censored_mask] - log_scale[censored_mask]))
            )
        
        # Add regularization terms
        # Penalize deviation from expected shape (around 2.0)
        shape_reg = 2.0 * torch.mean((shape - 2.0) ** 2)  # Increased from 0.5
        
        # Encourage larger scales when needed
        scale_reg = 0.05 * torch.mean(torch.relu(100.0 - scale))  # Reduced from 0.1
        
        # Add L1 regularization for sparsity
        l1_reg = 0.005 * (torch.mean(torch.abs(shape - 2.0)) + torch.mean(torch.abs(log_scale - np.log(100.0))))  # Reduced from 0.01
        
        return -(log_likelihood.mean() - shape_reg - scale_reg - l1_reg)


class WeibullRNNModel:
    """Implementation of Weibull Time-To-Event RNN model."""
    
    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        num_epochs: int = 50,
        patience: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        bidirectional: bool = True
    ):
        """Initialize the model.
        
        Args:
            input_size: Number of input features (optional, set during fit if not provided)
            hidden_size: Size of RNN hidden state
            num_layers: Number of RNN layers
            dropout: Dropout probability
            learning_rate: Initial learning rate
            batch_size: Batch size for training
            num_epochs: Maximum number of training epochs
            patience: Number of epochs to wait for improvement before early stopping
            device: Device to run the model on
            bidirectional: Whether to use bidirectional RNN
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device
        self.bidirectional = bidirectional
        
        self.model = None
        self.feature_names = None
        
        # Initialize model if input_size is provided
        if input_size is not None:
            self._initialize_model()
    
    def _initialize_model(self):
        self.model = WeibullRNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional
        ).to(self.device)
    
    def fit(self, data: WeibullRNNData) -> WeibullRNNResult:
        """Fit the model to training data with early stopping and learning rate scheduling."""
        # Create dataset
        dataset = WeibullRNNDataset(data)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Initialize model if not already initialized
        self.feature_names = dataset.feature_names
        if self.model is None:
            self.input_size = dataset.n_features
            self._initialize_model()
        
        # Initialize optimizer, scheduler, and loss
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01  # L2 regularization
        )
        
        # Cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,  # Restart every 20 epochs
            T_mult=2,  # Double the restart interval after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        
        criterion = WeibullLoss()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for sequences, times, events in train_loader:
                sequences = sequences.to(self.device)
                times = times.to(self.device)
                events = events.to(self.device)
                
                optimizer.zero_grad()
                shape, scale = self.model(sequences)
                loss = criterion(shape, scale, times, events)
                
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for sequences, times, events in val_loader:
                    sequences = sequences.to(self.device)
                    times = times.to(self.device)
                    events = events.to(self.device)
                    
                    shape, scale = self.model(sequences)
                    loss = criterion(shape, scale, times, events)
                    val_loss += loss.item()
            
            # Update learning rate
            scheduler.step()
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            all_sequences = torch.stack(dataset.sequences).to(self.device)
            shape, scale = self.model(all_sequences)
            
            # Calculate average parameters
            avg_shape = shape.mean().item()
            avg_scale = scale.mean().item()
            
            # Calculate average predicted time
            predicted_time = avg_scale * gamma(1 + 1/avg_shape)
            
            # Calculate survival probability at predicted time
            survival_prob = np.exp(
                -(predicted_time / avg_scale) ** avg_shape
            )
        
        # Create output
        model_output = WeibullRNNOutput(
            weibull_params=WeibullParameters(
                shape=avg_shape,
                scale=avg_scale
            ),
            predicted_time=predicted_time,
            survival_probability=survival_prob,
            log_likelihood=-best_val_loss
        )
        
        return WeibullRNNResult(model_metrics=model_output)
    
    def predict(self, data: WeibullRNNData) -> WeibullRNNResult:
        """Make predictions for new data.
        
        Args:
            data: Test data
        
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create dataset
        dataset = WeibullRNNDataset(data)
        
        # Make predictions
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i, (sequence, time, event) in enumerate(dataset):
                sequence = sequence.unsqueeze(0).to(self.device)
                shape, scale = self.model(sequence)
                
                # Calculate predicted time
                predicted_time = scale.item() * gamma(1 + 1/shape.item())
                
                # Calculate survival probability at predicted time
                survival_prob = np.exp(
                    -(predicted_time / scale.item()) ** shape.item()
                )
                
                predictions.append(
                    WeibullRNNPrediction(
                        entity_id=data.data[i].entity_id,
                        predicted_time=predicted_time,
                        survival_probability=survival_prob,
                        weibull_params=WeibullParameters(
                            shape=shape.item(),
                            scale=scale.item()
                        )
                    )
                )
        
        # Calculate average metrics
        avg_shape = np.mean([p.weibull_params.shape for p in predictions])
        avg_scale = np.mean([p.weibull_params.scale for p in predictions])
        avg_time = np.mean([p.predicted_time for p in predictions])
        avg_prob = np.mean([p.survival_probability for p in predictions])
        
        model_output = WeibullRNNOutput(
            weibull_params=WeibullParameters(
                shape=avg_shape,
                scale=avg_scale
            ),
            predicted_time=avg_time,
            survival_probability=avg_prob
        )
        
        return WeibullRNNResult(
            model_metrics=model_output,
            predictions=predictions
        )


if __name__ == "__main__":
    from generate import WeibullRNNConfig, generate_weibull_rnn_data
    from shared.distributions import WeibullParams, DistributionType
    
    # Example configuration
    config = WeibullRNNConfig(
        name="Test Config",
        survival_distribution=WeibullParams(
            distribution=DistributionType.WEIBULL,
            sample_size=1000,
            censoring_rate=0.3,
            shape=1.5,
            scale=100
        ),
        sequence_length=52,
        sequences=[
            SequenceConfig(
                name="logins_per_week",
                base_mean=5,
                base_std=1,
                trend_coefficient=-0.02,
                seasonality_amplitude=1,
                seasonality_period=52,
                noise_std=0.5,
                effect_on_survival=-0.5
            ),
            SequenceConfig(
                name="posts_per_week",
                base_mean=10,
                base_std=2,
                trend_coefficient=-0.05,
                seasonality_amplitude=2,
                seasonality_period=26,
                noise_std=1.0,
                effect_on_survival=-0.3
            ),
            SequenceConfig(
                name="native_posts_ratio",
                base_mean=0.3,
                base_std=0.1,
                trend_coefficient=0.01,
                seasonality_amplitude=0.1,
                seasonality_period=13,
                noise_std=0.05,
                effect_on_survival=0.7
            )
        ]
    )
    
    # Generate data
    data = generate_weibull_rnn_data(config)
    
    # Create and fit model
    model = WeibullRNNModel(
        hidden_size=128,
        num_layers=3,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=64,
        num_epochs=200,
        patience=10
    )
    result = model.fit(data)
    
    # Print results
    print("Model Results:")
    print(f"Shape Parameter: {result.model_metrics.weibull_params.shape:.4f}")
    print(f"Scale Parameter: {result.model_metrics.weibull_params.scale:.4f}")
    print(f"Predicted Time: {result.model_metrics.predicted_time:.4f}")
    print(f"Survival Probability: {result.model_metrics.survival_probability:.4f}")
    print(f"Log Likelihood: {result.model_metrics.log_likelihood:.4f}")
    
    # Make predictions
    predictions = model.predict(data)
    print("\nExample Predictions:")
    for pred in predictions.predictions[:5]:
        print(f"Entity {pred.entity_id}:")
        print(f"  Predicted Time: {pred.predicted_time:.4f}")
        print(f"  Survival Probability: {pred.survival_probability:.4f}")
        print(f"  Shape: {pred.weibull_params.shape:.4f}")
        print(f"  Scale: {pred.weibull_params.scale:.4f}") 