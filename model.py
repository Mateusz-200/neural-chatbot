import torch
import torch.nn as nn
import torch.optim as optim

class NeuralChatbot(nn.Module):
    """Neural network chatbot with learning capabilities"""
    
    def __init__(self, input_size, hidden_size=128, output_size=10):
        super(NeuralChatbot, self).__init__()
        self.hidden_size = hidden_size
        
        # Neural network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, x):
        """Forward pass through network"""
        x = x.to(self.device)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
    def train_on_batch(self, input_tensor, target_tensor, learning_rate=0.01):
        """Train on a single batch"""
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        output = self.forward(input_tensor)
        loss = loss_fn(output, target_tensor)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def predict(self, input_tensor):
        """Make prediction"""
        with torch.no_grad():
            output = self.forward(input_tensor)
            prediction = torch.argmax(output, dim=1)
        return prediction.item(), output.max().item()
