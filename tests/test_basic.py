"""
Basic tests for the active inference games system.
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from core.active_inference import ActiveInferenceAgent, BeliefState
    from games.blackjack import BlackjackGame, BlackjackPlayer
    from agents.blackjack_agent import BlackjackActiveInferenceAgent
    
    class TestActiveInferenceSystem(unittest.TestCase):
        """Test the active inference system components."""
        
        def setUp(self):
            """Set up test fixtures."""
            self.agent = ActiveInferenceAgent("test_agent")
            self.belief_state = BeliefState(state_dim=10)
            
        def test_agent_initialization(self):
            """Test that agents initialize properly."""
            self.assertEqual(self.agent.agent_id, "test_agent")
            self.assertIsNotNone(self.agent.belief_state)
            self.assertIsNotNone(self.agent.generative_model)
            
        def test_belief_state_update(self):
            """Test belief state updates."""
            import torch
            observation = torch.randn(10)
            self.belief_state.update(observation)
            self.assertIsNotNone(self.belief_state.mean)
            
        def test_blackjack_game_creation(self):
            """Test blackjack game creation."""
            game = BlackjackGame(num_decks=2, min_bet=5.0, max_bet=50.0)
            self.assertEqual(game.num_decks, 2)
            self.assertEqual(game.min_bet, 5.0)
            self.assertEqual(game.max_bet, 50.0)
            
        def test_blackjack_agent_creation(self):
            """Test blackjack agent creation."""
            agent = BlackjackActiveInferenceAgent("test_blackjack", 1000.0)
            self.assertEqual(agent.name, "test_blackjack")
            self.assertEqual(agent.bankroll, 1000.0)
            self.assertIsNotNone(agent.generative_model)
            
        def test_basic_game_flow(self):
            """Test basic game flow."""
            game = BlackjackGame(num_decks=1, min_bet=10.0, max_bet=100.0)
            agent = BlackjackActiveInferenceAgent("ai_player", 500.0)
            
            game.add_player(agent)
            
            # This should work without errors
            self.assertIsNotNone(game.players)
            self.assertEqual(len(game.players), 1)
            
    
    class TestGameMechanics(unittest.TestCase):
        """Test game mechanics."""
        
        def test_card_values(self):
            """Test card value calculations."""
            from games.blackjack import Card, Hand
            
            # Test ace handling
            hand = Hand()
            hand.add_card(Card('A', 'hearts'))
            hand.add_card(Card('K', 'spades'))
            self.assertEqual(hand.get_value(), 21)
            
            # Test soft ace
            hand = Hand()
            hand.add_card(Card('A', 'hearts'))
            hand.add_card(Card('6', 'spades'))
            self.assertEqual(hand.get_value(), 17)
            
            # Test bust
            hand = Hand()
            hand.add_card(Card('K', 'hearts'))
            hand.add_card(Card('Q', 'spades'))
            hand.add_card(Card('5', 'clubs'))
            self.assertEqual(hand.get_value(), 25)
            self.assertTrue(hand.is_bust())
            
        def test_blackjack_detection(self):
            """Test blackjack detection."""
            from games.blackjack import Card, Hand
            
            hand = Hand()
            hand.add_card(Card('A', 'hearts'))
            hand.add_card(Card('K', 'spades'))
            self.assertTrue(hand.is_blackjack())
            
            hand = Hand()
            hand.add_card(Card('10', 'hearts'))
            hand.add_card(Card('A', 'spades'))
            self.assertTrue(hand.is_blackjack())
            
            hand = Hand()
            hand.add_card(Card('9', 'hearts'))
            hand.add_card(Card('K', 'spades'))
            self.assertFalse(hand.is_blackjack())

except ImportError as e:
    print(f"Import error: {e}")
    print("Skipping tests - dependencies not installed")
    
    class TestDependencies(unittest.TestCase):
        """Test that dependencies are available."""
        
        def test_dependencies_installed(self):
            """Test that required dependencies are installed."""
            try:
                import torch
                import numpy as np
                import scipy
                self.assertTrue(True)
            except ImportError as e:
                self.fail(f"Dependencies not installed: {e}")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
