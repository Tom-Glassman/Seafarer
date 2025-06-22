import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import math

class InventoryNeuralNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.4):
        super(InventoryNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        layers = []
        prev_size = input_size
        
        # Setting layers to each have a linear layer, ReLU activation, batch norm, and dropout
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        self.initialize_weights()
    
    """
    Setter for starting vals for weights 
    """
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                input = module.weight.size(1)
                output = module.weight.size(0)
                
                bound = math.sqrt(6.0 / (input + output))
                
                with torch.no_grad():
                    module.weight.uniform_(-bound, bound)
                    module.bias.fill_(0.0)
    
    def forward(self, x):
        return self.network(x)

class NeuralNetworkTrainer:
    
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001, device=None):
        # Setting device to cpu
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = InventoryNeuralNetwork(input_size, hidden_sizes, output_size)
        self.model.to(self.device)
        
        # Using MSE loss and the Adam optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.7, 
            patience=10
        )
        
        self.train_losses = []
        self.val_losses = []
    
    """
    Splits the data into training and validation sets
    """
    def prepare_data(self, X, y, validation_split=0.2, batch_size=32):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        # Spliting validation and training
        split = int(len(X) * (1 - validation_split))
        training_indices = indices[:split]
        validation_indices = indices[split:]

        X_training = X[training_indices]
        X_validation = X[validation_indices]

        y_training = y[training_indices]
        y_validaton = y[validation_indices]
        
        X_training_tensor = torch.FloatTensor(X_training)
        X_val_tensor = torch.FloatTensor(X_validation)

        y_training_tensor = torch.FloatTensor(y_training)
        y_validation_tensor = torch.FloatTensor(y_validaton)
        
        training_dataset = TensorDataset(X_training_tensor, y_training_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_validation_tensor)
        
        self.training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        self.validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return self.training_loader, self.validation_loader
    
    """
    Single pass thru training data and updates weights
    """
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_X, batch_y in self.training_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            # Reset gradient
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            
            # Limiting gradients so they dont explode
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    """
    Checks to see training performace
    """
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            # Validation data loop
            for batch_X, batch_y in self.validation_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    """
    Trains the model
    """
    def train(self, X, y, epochs=500, validation_split=0.2, batch_size=32, patience=50):
        self.prepare_data(X, y, validation_split, batch_size)
        
        top_val = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            if epoch % 50 == 0:
                print(f"Epoch:  {epoch}")
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < top_val:
                top_val = val_loss
                patience_counter = 0
                self.top_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break
        
        if hasattr(self, 'top_state'):
            self.model.load_state_dict(self.top_state)
        
        return self.train_losses, self.val_losses
    
    """
    Makes a prediction based on the model
    """
    def make_prediction(self, X):
        self.model.eval()
        
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X
        
        X_tensor = X_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()

"""
Interface for the neural network.  Started w/ simple but had to expand
"""
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        self.trainer = NeuralNetworkTrainer(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            learning_rate=learning_rate
        )
    
    """
    Wrapper that calls the other training method and returns only training loss
    """
    def train(self, X, y, epochs=500):
        training_losses, validation_losses = self.trainer.train(
            X, y, epochs = epochs, patience = 50
        )
        return training_losses
    
    """
    Wrapper for predictions
    """
    def make_prediction(self, X):
        return self.trainer.make_prediction(X)
    
    """
    Wrapper for forward pass
    """
    def forward(self, X):
        return self.make_prediction(X)