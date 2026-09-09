import numpy as np
import torch
import pickle
import os

class TextEncoder:
    """Encode text to numerical vectors"""
    
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        words = set()
        for text in texts:
            words.update(text.lower().split())
        
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(words))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
    def encode(self, text):
        """Convert text to numerical vector"""
        vector = np.zeros(self.vocab_size)
        for word in text.lower().split():
            if word in self.word_to_idx:
                vector[self.word_to_idx[word]] += 1
        return vector
    
    def decode(self, vector):
        """Convert vector back to text"""
        words = []
        for idx, count in enumerate(vector):
            if count > 0 and idx in self.idx_to_word:
                words.extend([self.idx_to_word[idx]] * int(count))
        return ' '.join(words)

class ModelManager:
    """Save and load models"""
    
    @staticmethod
    def save_model(model, encoder, filename='chatbot_model.pkl'):
        """Save model and encoder"""
        data = {
            'model_state': model.state_dict(),
            'encoder': encoder
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {filename}")
    
    @staticmethod
    def load_model(model, filename='chatbot_model.pkl'):
        """Load model and encoder"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            model.load_state_dict(data['model_state'])
            encoder = data['encoder']
            print(f"Model loaded from {filename}")
            return encoder
        return None

def create_training_data(sentences, labels):
    """Create training tensors"""
    encoder = TextEncoder()
    encoder.build_vocab(sentences)
    
    X = []
    y = []
    for sentence, label in zip(sentences, labels):
        X.append(encoder.encode(sentence))
        y.append(label)
    
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    return X, y, encoder
