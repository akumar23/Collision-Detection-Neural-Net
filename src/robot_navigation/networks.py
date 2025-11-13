import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, output_size=1):
        super(Action_Conditioned_FF, self).__init__()
        # Increased hidden size for better capacity
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size // 2)  # Add another layer
        self.nonlinear_activation = nn.ReLU()  # ReLU is better than Sigmoid for hidden layers
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization
        self.hidden_to_output = nn.Linear(hidden_size // 2, output_size)
        # Note: No sigmoid on output - BCEWithLogitsLoss includes sigmoid

    def forward(self, input):
        hidden = self.input_to_hidden(input)
        hidden = self.nonlinear_activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.hidden_to_hidden(hidden)
        hidden = self.nonlinear_activation(hidden)
        output = self.hidden_to_output(hidden)
        return output

    def evaluate(self, model, test_loader, loss_function):
        model.eval()
        total_loss = 0
        num_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(test_loader):
                output = self.forward(sample['input'])
                loss = loss_function(output, sample['label'])
                total_loss += loss.item()
                num_samples += 1
        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        return avg_loss

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
