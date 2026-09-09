import torch
import random
from model import NeuralChatbot
from utils import TextEncoder, create_training_data, ModelManager
from games import WordPredictionGame, MathGame, QuizGame

class Chatbot:
    """Main chatbot class with learning capabilities"""
    
    def __init__(self):
        self.encoder = TextEncoder()
        self.model = None
        self.conversation_history = []
        self.responses = {
            0: "Hello! How can I help you?",
            1: "That's interesting!",
            2: "Tell me more about that.",
            3: "I see what you mean.",
            4: "Can you explain that further?",
            5: "I'm learning from our conversation!",
            6: "That sounds good!",
            7: "Interesting point!",
            8: "Keep teaching me!",
            9: "Thanks for chatting!"
        }
        
        self._initialize_vocab()
        self.model = NeuralChatbot(
            input_size=len(self.encoder.word_to_idx) if self.encoder.vocab_size > 0 else 100,
            hidden_size=128,
            output_size=len(self.responses)
        )
        
        self.games = {
            '1': WordPredictionGame(self.model, self.encoder),
            '2': MathGame(self.model, self.encoder),
            '3': QuizGame(self.model, self.encoder)
        }
    
    def _initialize_vocab(self):
        """Initialize vocabulary with common words"""
        all_texts = list(self.responses.values()) + [
            "hello world", "good morning", "how are you",
            "what is your name", "tell me a joke", "help me",
            "thank you", "goodbye", "see you later", "nice to meet you"
        ]
        self.encoder.build_vocab(all_texts)
    
    def chat(self, user_input):
        """Generate response to user input"""
        self.conversation_history.append(user_input)
        
        try:
            # Encode input
            input_vector = self.encoder.encode(user_input)
            input_tensor = torch.FloatTensor(input_vector).unsqueeze(0)
            
            # Get prediction
            prediction, confidence = self.model.predict(input_tensor)
            
            # Get response
            response = self.responses[prediction]
            
            # Train on this interaction (reinforcement)
            target = torch.LongTensor([prediction])
            self.model.train_on_batch(input_tensor, target, learning_rate=0.001)
            
            return response, confidence
        except Exception as e:
            return "I'm still learning, please try again!", 0.5
    
    def play_game(self, game_choice):
        """Play a training game"""
        if game_choice not in self.games:
            print("Invalid game choice!")
            return
        
        game = self.games[game_choice]
        game.play()
    
    def save(self, filename='chatbot_model.pkl'):
        """Save trained model"""
        ModelManager.save_model(self.model, self.encoder, filename)
    
    def load(self, filename='chatbot_model.pkl'):
        """Load trained model"""
        encoder = ModelManager.load_model(self.model, filename)
        if encoder:
            self.encoder = encoder
    
    def show_stats(self):
        """Show chatbot statistics"""
        print("\n📊 CHATBOT STATISTICS")
        print("=" * 40)
        print(f"Vocabulary size: {self.encoder.vocab_size}")
        print(f"Conversation history: {len(self.conversation_history)} messages")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"Recent messages:")
        for msg in self.conversation_history[-5:]:
            print(f"  - {msg}")
