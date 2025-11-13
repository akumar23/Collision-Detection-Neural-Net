import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot_navigation.data_loaders import Data_Loaders
from src.robot_navigation.networks import Action_Conditioned_FF

import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):
    batch_size = 32
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()

    losses = []
    # Use BCEWithLogitsLoss for binary classification (collision/no collision)
    loss_function = nn.BCEWithLogitsLoss()
    learning_rate = 0.01  # Lower learning rate for more stable training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer is better than SGD
    
    # Evaluate initial model
    model.eval()
    initial_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(initial_loss)
    print(f"Initial test loss: {initial_loss:.4f}")

    for epoch_i in range(no_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        # Use train_loader for training (not test_loader!)
        for idx, sample in enumerate(data_loaders.train_loader):
            optimizer.zero_grad()
            output = model(sample['input'])
            loss = loss_function(output, sample['label'])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        # Evaluate on test set
        model.eval()
        test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        losses.append(test_loss)
        
        if (epoch_i + 1) % 10 == 0 or epoch_i == 0:
            print(f"Epoch {epoch_i+1}/{no_epochs} - Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Save model after each epoch
        models_path = Path(__file__).parent.parent / "models" / "saved_model.pkl"
        torch.serialization.save(model.state_dict(), models_path)
    
    print(f"Training complete. Final test loss: {test_loss:.4f}")
    return losses


if __name__ == '__main__':
    no_epochs = 100  # Train for more epochs for better accuracy
    start = time.time()
    train_model(no_epochs)
    end = time.time()
    print(f"Total training time: {end - start:.2f} seconds")
