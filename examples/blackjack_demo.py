"""
Blackjack Demo - Active Inference Agent vs Basic Strategy

This demo shows how to use the active inference agent to play blackjack,
comparing it against a basic strategy player and a human player.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# Import our classes (will work once dependencies are installed)
try:
    from agents.blackjack_agent import BlackjackActiveInferenceAgent
    from games.blackjack import BlackjackGame, BlackjackPlayer, BlackjackAction, BlackjackGameState
    
    class BasicStrategyPlayer(BlackjackPlayer):
        """Simple basic strategy player for comparison."""
        
        def __init__(self, name: str, bankroll: float):
            super().__init__(name, bankroll)
        
        def make_bet(self, min_bet: float, max_bet: float) -> float:
            return min_bet
        
        def choose_action(self, game_state: BlackjackGameState) -> BlackjackAction:
            return game_state.game.get_basic_strategy_action(
                game_state.player_hands[game_state.current_hand]
            )
        
        def insurance_decision(self, game_state: BlackjackGameState) -> bool:
            return False  # Never take insurance
    
    
    class HumanPlayer(BlackjackPlayer):
        """Human player with console interface."""
        
        def __init__(self, name: str, bankroll: float):
            super().__init__(name, bankroll)
        
        def make_bet(self, min_bet: float, max_bet: float) -> float:
            while True:
                try:
                    bet = float(input(f"Enter your bet (${min_bet:.2f} - ${max_bet:.2f}): $"))
                    if min_bet <= bet <= max_bet and bet <= self.bankroll:
                        return bet
                    else:
                        print(f"Invalid bet. Must be between ${min_bet:.2f} and ${max_bet:.2f}")
                except ValueError:
                    print("Please enter a valid number")
        
        def choose_action(self, game_state: BlackjackGameState) -> BlackjackAction:
            print(f"\nYour hand: {game_state.player_hands[game_state.current_hand]}")
            print(f"Dealer up card: {game_state.dealer_hand.cards[0]}")
            
            actions = ["hit", "stand"]
            current_hand = game_state.player_hands[game_state.current_hand]
            
            if current_hand.can_double():
                actions.append("double")
            if current_hand.can_split():
                actions.append("split")
            if len(current_hand.cards) == 2:
                actions.append("surrender")
            
            while True:
                action_str = input(f"Choose action ({'/'.join(actions)}): ").lower()
                if action_str in actions:
                    return BlackjackAction(action_str)
                print("Invalid action. Please try again.")
        
        def insurance_decision(self, game_state: BlackjackGameState) -> bool:
            while True:
                decision = input("Take insurance? (y/n): ").lower()
                if decision in ['y', 'yes']:
                    return True
                elif decision in ['n', 'no']:
                    return False
                print("Please enter 'y' or 'n'")
    
    
    def run_simulation(num_games: int = 1000) -> Dict[str, List[float]]:
        """Run a simulation comparing different player types."""
        print(f"Running {num_games} game simulation...")
        
        # Create players
        ai_agent = BlackjackActiveInferenceAgent("AI_Agent", 1000.0, risk_tolerance=0.3)
        basic_player = BasicStrategyPlayer("Basic_Strategy", 1000.0)
        
        # Create game
        game = BlackjackGame(num_decks=6, min_bet=5.0, max_bet=100.0)
        game.add_player(ai_agent)
        game.add_player(basic_player)
        
        # Track results
        results = {
            "AI_Agent": [],
            "Basic_Strategy": []
        }
        
        bankrolls = {
            "AI_Agent": [1000.0],
            "Basic_Strategy": [1000.0]
        }
        
        for game_num in range(num_games):
            # Play a round
            round_results = game.play_round()
            
            # Update tracking
            for player_name, winnings in round_results.items():
                results[player_name].append(winnings)
                current_bankroll = bankrolls[player_name][-1] + winnings
                bankrolls[player_name].append(current_bankroll)
            
            # Let AI agent learn from results
            if "AI_Agent" in round_results:
                ai_agent.learn_from_result(round_results["AI_Agent"], game.game_state)
            
            # Print progress
            if (game_num + 1) % 100 == 0:
                print(f"Game {game_num + 1}/{num_games}")
                print(f"AI Agent: ${ai_agent.bankroll:.2f}")
                print(f"Basic Strategy: ${basic_player.bankroll:.2f}")
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Bankroll evolution
        plt.subplot(2, 2, 1)
        plt.plot(bankrolls["AI_Agent"], label="AI Agent", color='blue')
        plt.plot(bankrolls["Basic_Strategy"], label="Basic Strategy", color='red')
        plt.xlabel('Game Number')
        plt.ylabel('Bankroll ($)')
        plt.title('Bankroll Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Cumulative winnings
        plt.subplot(2, 2, 2)
        ai_cumulative = np.cumsum(results["AI_Agent"])
        basic_cumulative = np.cumsum(results["Basic_Strategy"])
        plt.plot(ai_cumulative, label="AI Agent", color='blue')
        plt.plot(basic_cumulative, label="Basic Strategy", color='red')
        plt.xlabel('Game Number')
        plt.ylabel('Cumulative Winnings ($)')
        plt.title('Cumulative Winnings')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Win rate distribution
        plt.subplot(2, 2, 3)
        ai_wins = np.array([1 if x > 0 else 0 for x in results["AI_Agent"]])
        basic_wins = np.array([1 if x > 0 else 0 for x in results["Basic_Strategy"]])
        
        window_size = 100
        ai_win_rate = np.convolve(ai_wins, np.ones(window_size)/window_size, mode='valid')
        basic_win_rate = np.convolve(basic_wins, np.ones(window_size)/window_size, mode='valid')
        
        plt.plot(ai_win_rate, label="AI Agent", color='blue')
        plt.plot(basic_win_rate, label="Basic Strategy", color='red')
        plt.xlabel('Game Number')
        plt.ylabel('Win Rate (100-game window)')
        plt.title('Win Rate Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Final statistics
        plt.subplot(2, 2, 4)
        stats = {
            'AI Agent': {
                'Final Bankroll': ai_agent.bankroll,
                'Total Winnings': sum(results["AI_Agent"]),
                'Win Rate': np.mean(ai_wins),
                'Avg Bet': np.mean([abs(x) for x in results["AI_Agent"] if x != 0])
            },
            'Basic Strategy': {
                'Final Bankroll': basic_player.bankroll,
                'Total Winnings': sum(results["Basic_Strategy"]),
                'Win Rate': np.mean(basic_wins),
                'Avg Bet': np.mean([abs(x) for x in results["Basic_Strategy"] if x != 0])
            }
        }
        
        # Display statistics as text
        plt.text(0.1, 0.8, "Final Statistics:", fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
        
        y_pos = 0.7
        for player, player_stats in stats.items():
            plt.text(0.1, y_pos, f"{player}:", fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
            y_pos -= 0.05
            for stat, value in player_stats.items():
                if isinstance(value, float):
                    plt.text(0.15, y_pos, f"{stat}: ${value:.2f}" if 'Bankroll' in stat or 'Winnings' in stat or 'Bet' in stat else f"{stat}: {value:.3f}", 
                            fontsize=10, transform=plt.gca().transAxes)
                else:
                    plt.text(0.15, y_pos, f"{stat}: {value}", fontsize=10, transform=plt.gca().transAxes)
                y_pos -= 0.04
            y_pos -= 0.02
        
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('blackjack_simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    
    
    def play_interactive_game():
        """Play an interactive game against the AI agent."""
        print("Welcome to Blackjack!")
        print("You're playing against an Active Inference AI agent.")
        
        # Create players
        human = HumanPlayer("Human", 1000.0)
        ai_agent = BlackjackActiveInferenceAgent("AI_Agent", 1000.0, risk_tolerance=0.2)
        
        # Create game
        game = BlackjackGame(num_decks=6, min_bet=5.0, max_bet=100.0)
        game.add_player(human)
        game.add_player(ai_agent)
        
        game_count = 0
        while human.bankroll > 0 and ai_agent.bankroll > 0:
            game_count += 1
            print(f"\n{'='*50}")
            print(f"GAME {game_count}")
            print(f"{'='*50}")
            
            print(f"Your bankroll: ${human.bankroll:.2f}")
            print(f"AI bankroll: ${ai_agent.bankroll:.2f}")
            
            # Play round
            try:
                results = game.play_round()
                
                # Show results
                print(f"\nRound Results:")
                for player_name, winnings in results.items():
                    print(f"{player_name}: ${winnings:+.2f}")
                
                # Let AI learn
                if "AI_Agent" in results:
                    ai_agent.learn_from_result(results["AI_Agent"], game.game_state)
                
                # Ask to continue
                if human.bankroll > 0 and ai_agent.bankroll > 0:
                    continue_game = input("\nContinue playing? (y/n): ").lower()
                    if continue_game not in ['y', 'yes']:
                        break
                
            except KeyboardInterrupt:
                print("\nGame interrupted by user.")
                break
            except Exception as e:
                print(f"Error during game: {e}")
                break
        
        # Final results
        print(f"\n{'='*50}")
        print("FINAL RESULTS")
        print(f"{'='*50}")
        print(f"Human final bankroll: ${human.bankroll:.2f}")
        print(f"AI final bankroll: ${ai_agent.bankroll:.2f}")
        
        if human.bankroll > ai_agent.bankroll:
            print("Congratulations! You beat the AI!")
        elif ai_agent.bankroll > human.bankroll:
            print("The AI agent won this session.")
        else:
            print("It's a tie!")
    
    
    def main():
        """Main function to run demos."""
        print("Active Inference Blackjack Demo")
        print("="*40)
        
        while True:
            print("\nChoose an option:")
            print("1. Run simulation (AI vs Basic Strategy)")
            print("2. Play interactive game (Human vs AI)")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                try:
                    num_games = int(input("Enter number of games to simulate (default 1000): ") or "1000")
                    run_simulation(num_games)
                except ValueError:
                    print("Invalid number. Using default 1000 games.")
                    run_simulation(1000)
                except Exception as e:
                    print(f"Error during simulation: {e}")
            
            elif choice == '2':
                try:
                    play_interactive_game()
                except Exception as e:
                    print(f"Error during interactive game: {e}")
            
            elif choice == '3':
                print("Thanks for playing!")
                break
            
            else:
                print("Invalid choice. Please try again.")

except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required dependencies:")
    print("pip install torch numpy matplotlib scipy")
    print("\nOr run: pip install -r requirements.txt")
    
    def main():
        print("Dependencies not installed. Please install them first.")


if __name__ == "__main__":
    main()
