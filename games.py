import random
import torch
from utils import TextEncoder

class TrainingGame:
    """Base class for training games"""
    
    def __init__(self, chatbot, encoder):
        self.chatbot = chatbot
        self.encoder = encoder
        self.score = 0
        
    def play(self):
        raise NotImplementedError

class WordPredictionGame(TrainingGame):
    """Game: Predict next word"""
    
    def __init__(self, chatbot, encoder):
        super().__init__(chatbot, encoder)
        self.word_pairs = [
            ("hello", "world"),
            ("good", "morning"),
            ("how", "are"),
            ("nice", "day"),
            ("thank", "you"),
            ("see", "you"),
            ("good", "bye"),
            ("take", "care"),
        ]
    
    def play(self, rounds=5):
        print("\n🎮 WORD PREDICTION GAME")
        print("=" * 40)
        self.score = 0
        
        for i in range(rounds):
            word, answer = random.choice(self.word_pairs)
            print(f"\nRound {i+1}: What word follows '{word}'?")
            
            user_input = input("Your answer: ").lower().strip()
            
            if user_input == answer:
                print("✓ Correct!")
                self.score += 1
                # Train the model
                self._train(f"{word} {answer}", 1)
            else:
                print(f"✗ Wrong! The answer was '{answer}'")
                self._train(f"{word} {answer}", 1)
        
        print(f"\nGame Over! Score: {self.score}/{rounds}")
        return self.score
    
    def _train(self, text, label):
        """Train model on game result"""
        try:
            input_tensor = torch.FloatTensor(self.encoder.encode(text)).unsqueeze(0)
            target_tensor = torch.LongTensor([label])
            loss = self.chatbot.train_on_batch(input_tensor, target_tensor)
        except:
            pass

class MathGame(TrainingGame):
    """Game: Solve math problems"""
    
    def __init__(self, chatbot, encoder):
        super().__init__(chatbot, encoder)
    
    def play(self, rounds=5):
        print("\n🎮 MATH GAME")
        print("=" * 40)
        self.score = 0
        
        for i in range(rounds):
            a = random.randint(1, 20)
            b = random.randint(1, 20)
            answer = a + b
            
            print(f"\nRound {i+1}: What is {a} + {b}?")
            
            try:
                user_input = int(input("Your answer: "))
                
                if user_input == answer:
                    print("✓ Correct!")
                    self.score += 1
                    self._train(f"plus math", 1)
                else:
                    print(f"✗ Wrong! The answer is {answer}")
                    self._train(f"plus math", 0)
            except ValueError:
                print("Invalid input!")
        
        print(f"\nGame Over! Score: {self.score}/{rounds}")
        return self.score
    
    def _train(self, text, label):
        """Train model on game result"""
        try:
            input_tensor = torch.FloatTensor(self.encoder.encode(text)).unsqueeze(0)
            target_tensor = torch.LongTensor([label])
            loss = self.chatbot.train_on_batch(input_tensor, target_tensor)
        except:
            pass

class QuizGame(TrainingGame):
    """Game: Answer trivia questions"""
    
    def __init__(self, chatbot, encoder):
        super().__init__(chatbot, encoder)
        self.questions = [
            {"q": "What is the capital of France?", "a": "paris"},
            {"q": "What is 2+2?", "a": "4"},
            {"q": "What color is the sky?", "a": "blue"},
            {"q": "What is the largest planet?", "a": "jupiter"},
            {"q": "What is the smallest country?", "a": "vatican"},
        ]
    
    def play(self, rounds=5):
        print("\n🎮 QUIZ GAME")
        print("=" * 40)
        self.score = 0
        
        for i in range(min(rounds, len(self.questions))):
            q_data = self.questions[i]
            print(f"\nRound {i+1}: {q_data['q']}")
            
            user_input = input("Your answer: ").lower().strip()
            
            if user_input == q_data['a']:
                print("✓ Correct!")
                self.score += 1
                self._train(q_data['q'], 1)
            else:
                print(f"✗ Wrong! The answer is '{q_data['a']}'")
                self._train(q_data['q'], 0)
        
        print(f"\nGame Over! Score: {self.score}/{min(rounds, len(self.questions))}")
        return self.score
    
    def _train(self, text, label):
        """Train model on game result"""
        try:
            input_tensor = torch.FloatTensor(self.encoder.encode(text)).unsqueeze(0)
            target_tensor = torch.LongTensor([label])
            loss = self.chatbot.train_on_batch(input_tensor, target_tensor)
        except:
            pass
