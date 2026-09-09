# 🤖 Neural Network Chatbot

A Python-based chatbot powered by a neural network that **learns from games and conversations**.

## Features

✨ **Neural Network Learning**: The chatbot uses a multi-layer neural network with backpropagation
🎮 **Training Games**: 3 different games to train the bot:
  - Word Prediction Game
  - Math Game
  - Quiz Game
💬 **Conversational AI**: Chat with the bot while it learns
📊 **Model Management**: Save and load trained models
🧠 **Reinforcement Learning**: The bot learns from every interaction

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Mateusz-200/neural-chatbot.git
cd neural-chatbot

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the Chatbot

```bash
python main.py
```

### Main Menu Options

1. **Chat with the bot**: Have a conversation while the bot learns
2. **Play training games**: Train the bot through interactive games
3. **View statistics**: See chatbot performance metrics
4. **Save model**: Save the trained neural network
5. **Load model**: Load a previously trained model
6. **Exit**: Close the application

## How It Works

### Neural Network Architecture

```
Input Layer (Vocabulary Size)
    ↓
Hidden Layer 1 (128 neurons, ReLU activation)
    ↓
Hidden Layer 2 (128 neurons, ReLU activation)
    ↓
Output Layer (10 response classes, Softmax)
```

### Learning Process

1. **Text Encoding**: User input is converted to numerical vectors using Bag-of-Words
2. **Forward Pass**: Vector goes through neural network
3. **Prediction**: Network predicts best response
4. **Backpropagation**: Network learns from feedback via games
5. **Model Update**: Weights are adjusted using Adam optimizer

### Training Games

#### Word Prediction Game
- Bot learns word associations
- Example: "hello" → "world"
- Improves contextual understanding

#### Math Game
- Bot learns simple arithmetic
- Improves numerical reasoning
- Example: "5 + 3 = 8"

#### Quiz Game
- Bot learns factual information
- Improves knowledge base
- Example: "Capital of France = Paris"

## File Structure

```
neural-chatbot/
├── main.py              # Main application entry point
├── chatbot.py           # Core chatbot class
├── model.py             # Neural network model definition
├── games.py             # Training games
├── utils.py             # Utility functions (encoding, model management)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Example Session

```
🤖 NEURAL NETWORK CHATBOT
==================================================
1. Chat with the bot
2. Play training games
3. View statistics
4. Save model
5. Load model
6. Exit
==================================================
Choose an option (1-6): 1

💬 CHAT MODE
(Type 'back' to return to main menu)
--------------------------------------------------

You: Hello!
Bot: Hello! How can I help you? (confidence: 0.87)

You: Tell me something interesting
Bot: That's interesting! (confidence: 0.79)

You: back

Choose an option (1-6): 2

🎮 SELECT A TRAINING GAME
==================================================
1. Word Prediction Game
2. Math Game
3. Quiz Game
4. Back to main menu
==================================================
Choose a game (1-4): 1

🎮 WORD PREDICTION GAME
========================================

Round 1: What word follows 'hello'?
Your answer: world
✓ Correct!

Game Over! Score: 5/5
```

## Model Persistence

Save your trained model:
```python
chatbot.save('my_model.pkl')
```

Load a previously trained model:
```python
chatbot.load('my_model.pkl')
```

## Customization

### Add More Training Games

Edit `games.py` to add custom games:

```python
class MyCustomGame(TrainingGame):
    def play(self):
        # Your game logic here
        pass
```

### Add More Responses

Edit `chatbot.py` responses dictionary:

```python
self.responses = {
    0: "Your custom response 1",
    1: "Your custom response 2",
    # ...
}
```

### Adjust Neural Network

Modify `model.py` parameters:

```python
self.model = NeuralChatbot(
    input_size=100,      # Vocabulary size
    hidden_size=256,     # Larger = more capacity
    output_size=10       # Number of response types
)
```

## Technical Details

- **Framework**: PyTorch
- **Optimization**: Adam optimizer
- **Loss Function**: CrossEntropyLoss
- **Encoding**: Bag-of-Words (BoW)
- **Device**: GPU (if available) / CPU

## Future Improvements

- [ ] LSTM/GRU for better context understanding
- [ ] Word embeddings (Word2Vec, GloVe)
- [ ] Attention mechanism
- [ ] Multi-turn conversation memory
- [ ] Natural Language Processing (NLP) improvements
- [ ] Transfer learning from pre-trained models
- [ ] Web interface (Flask/Django)

## License

MIT License - Feel free to use and modify!

## Author

Created with ❤️ for learning AI and neural networks

---

**Happy learning! 🚀**
