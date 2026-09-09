#!/usr/bin/env python3
"""
Neural Network Chatbot - Main Application
A chatbot that learns through games and conversations
"""

from chatbot import Chatbot
import os

def clear_screen():
    """Clear console"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_menu():
    """Print main menu"""
    print("\n" + "="*50)
    print("🤖 NEURAL NETWORK CHATBOT")
    print("="*50)
    print("1. Chat with the bot")
    print("2. Play training games")
    print("3. View statistics")
    print("4. Save model")
    print("5. Load model")
    print("6. Exit")
    print("="*50)

def print_game_menu():
    """Print game menu"""
    print("\n" + "="*50)
    print("🎮 SELECT A TRAINING GAME")
    print("="*50)
    print("1. Word Prediction Game")
    print("2. Math Game")
    print("3. Quiz Game")
    print("4. Back to main menu")
    print("="*50)

def main():
    """Main application loop"""
    clear_screen()
    print("\n🚀 Initializing Neural Network Chatbot...")
    chatbot = Chatbot()
    print("✓ Chatbot initialized and ready to learn!")
    
    while True:
        print_menu()
        choice = input("Choose an option (1-6): ").strip()
        
        if choice == '1':
            print("\n💬 CHAT MODE")
            print("(Type 'back' to return to main menu)")
            print("-" * 50)
            while True:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'back':
                    break
                
                if not user_input:
                    continue
                
                response, confidence = chatbot.chat(user_input)
                print(f"Bot: {response} (confidence: {confidence:.2f})")
        
        elif choice == '2':
            while True:
                print_game_menu()
                game_choice = input("Choose a game (1-4): ").strip()
                
                if game_choice == '4':
                    break
                elif game_choice in ['1', '2', '3']:
                    chatbot.play_game(game_choice)
                    input("\nPress Enter to continue...")
                else:
                    print("Invalid choice!")
        
        elif choice == '3':
            chatbot.show_stats()
            input("\nPress Enter to continue...")
        
        elif choice == '4':
            filename = input("Enter filename to save (default: chatbot_model.pkl): ").strip()
            if not filename:
                filename = 'chatbot_model.pkl'
            chatbot.save(filename)
            input("Press Enter to continue...")
        
        elif choice == '5':
            filename = input("Enter filename to load (default: chatbot_model.pkl): ").strip()
            if not filename:
                filename = 'chatbot_model.pkl'
            chatbot.load(filename)
            input("Press Enter to continue...")
        
        elif choice == '6':
            print("\n👋 Thank you for using Neural Network Chatbot!")
            print("Goodbye! The bot learned from our interaction.")
            break
        
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Chatbot interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check the error and try again.")
