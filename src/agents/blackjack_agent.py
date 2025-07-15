"""
Active Inference Agent for Blackjack

Implements an active inference agent that can play blackjack using:
- Friston's free energy principle
- Bayesian belief updating
- Predictive processing
- Michael Levin's goal-directed behavior concepts
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ..inference.active_inference import ActiveInferenceAgent, LevinianAgent, Action, Observation
from ..games.blackjack import BlackjackPlayer, BlackjackAction, BlackjackGameState, Hand


@dataclass
class BlackjackBelief:
    """Beliefs specific to blackjack."""
    card_count_belief: float
    dealer_hole_card_belief: Dict[int, float]  # Probability distribution over dealer's hole card
    deck_composition_belief: Dict[int, int]  # Believed remaining cards of each rank
    win_probability: float
    expected_value: float


class BlackjackGenerativeModel(nn.Module):
    """
    Generative model for blackjack that predicts:
    1. Next card probabilities
    2. Dealer behavior
    3. Game outcomes
    """
    
    def __init__(self, input_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        
        # Card prediction network
        self.card_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 13)  # 13 possible card ranks
        )
        
        # Dealer behavior model
        self.dealer_model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Will hit or stand
        )
        
        # Outcome prediction
        self.outcome_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Win, lose, push
        )
        
        # Value network for expected utility
        self.value_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Expected value
        )
    
    def forward(self, game_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through all networks."""
        card_probs = torch.softmax(self.card_predictor(game_state), dim=-1)
        dealer_action_probs = torch.softmax(self.dealer_model(game_state), dim=-1)
        outcome_probs = torch.softmax(self.outcome_predictor(game_state), dim=-1)
        expected_value = self.value_network(game_state)
        
        return {
            'card_probs': card_probs,
            'dealer_action_probs': dealer_action_probs,
            'outcome_probs': outcome_probs,
            'expected_value': expected_value
        }


class BlackjackActiveInferenceAgent(LevinianAgent, BlackjackPlayer):
    """
    Active inference agent for blackjack that combines:
    - Friston's active inference framework
    - Michael Levin's goal-directed behavior
    - Blackjack-specific domain knowledge
    """
    
    def __init__(self, name: str, bankroll: float, risk_tolerance: float = 0.5):
        # Initialize BlackjackPlayer
        BlackjackPlayer.__init__(self, name, bankroll)
        
        # Initialize active inference components
        self.observation_dim = 16  # Game state features
        self.hidden_dim = 32
        self.action_dim = 5  # Number of possible actions
        
        LevinianAgent.__init__(self, self.observation_dim, self.hidden_dim, self.action_dim)
        
        # Blackjack-specific components
        self.blackjack_model = BlackjackGenerativeModel(self.observation_dim)
        self.risk_tolerance = risk_tolerance
        self.card_memory = {}  # Memory of seen cards
        self.betting_strategy = "flat"  # Can be "flat", "progressive", "kelly"
        
        # Blackjack beliefs
        self.blackjack_beliefs = BlackjackBelief(
            card_count_belief=0.0,
            dealer_hole_card_belief={i: 1/13 for i in range(1, 14)},
            deck_composition_belief={i: 4 for i in range(1, 14)},
            win_probability=0.5,
            expected_value=0.0
        )
        
        # Learning parameters
        self.learning_rate = 0.001
        self.blackjack_optimizer = torch.optim.Adam(
            self.blackjack_model.parameters(), 
            lr=self.learning_rate
        )
        
        # Goals (Levinian agent goals)
        self.goals = torch.tensor([
            1.0,  # Maximize winnings
            0.5,  # Minimize losses
            0.3,  # Learn optimal strategy
            0.2   # Maintain bankroll
        ])
    
    def game_state_to_observation(self, game_state: BlackjackGameState) -> Observation:
        """Convert blackjack game state to observation tensor."""
        features = []
        
        # Current hand information
        if game_state.current_hand < len(game_state.player_hands):
            current_hand = game_state.player_hands[game_state.current_hand]
            features.extend([
                current_hand.get_value() / 21.0,  # Normalized hand value
                1.0 if current_hand.is_soft() else 0.0,
                1.0 if current_hand.can_split() else 0.0,
                1.0 if current_hand.can_double() else 0.0,
                1.0 if current_hand.is_blackjack() else 0.0,
                len(current_hand.cards) / 10.0  # Normalized card count
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Dealer information
        if game_state.dealer_hand.cards:
            dealer_up_card = game_state.dealer_hand.cards[0].rank.value
            features.extend([
                dealer_up_card / 14.0,  # Normalized dealer up card
                1.0 if dealer_up_card == 11 else 0.0  # Ace up
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Deck information
        features.extend([
            game_state.deck.penetration(),  # Deck penetration
            game_state.true_count / 10.0,  # Normalized true count
            game_state.running_count / 50.0  # Normalized running count
        ])
        
        # Add more features to reach observation_dim
        while len(features) < self.observation_dim:
            features.append(0.0)
        
        # Truncate if too many features
        features = features[:self.observation_dim]
        
        return Observation(
            data=torch.tensor(features, dtype=torch.float32),
            timestamp=0.0,
            source="blackjack_game"
        )
    
    def update_beliefs_from_cards(self, cards_seen: List) -> None:
        """Update beliefs based on observed cards."""
        for card in cards_seen:
            if card not in self.card_memory:
                self.card_memory[card] = 0
            self.card_memory[card] += 1
            
            # Update deck composition belief
            rank_value = card.rank.value
            if rank_value in self.blackjack_beliefs.deck_composition_belief:
                self.blackjack_beliefs.deck_composition_belief[rank_value] -= 1
        
        # Update card count belief using Hi-Lo system
        card_values = {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1, 11: -1}
        
        for card in cards_seen:
            if card not in self.card_memory or self.card_memory[card] == 1:
                rank_value = min(card.rank.value, 11)  # Treat face cards as 10
                if rank_value in card_values:
                    self.blackjack_beliefs.card_count_belief += card_values[rank_value]
    
    def compute_expected_value(self, action: BlackjackAction, game_state: BlackjackGameState) -> float:
        """Compute expected value for an action using the generative model."""
        obs = self.game_state_to_observation(game_state)
        
        with torch.no_grad():
            model_output = self.blackjack_model(obs.data.unsqueeze(0))
            
            # Get win probability and expected value
            outcome_probs = model_output['outcome_probs'][0]
            expected_value = model_output['expected_value'][0].item()
            
            # Adjust based on action
            if action == BlackjackAction.HIT:
                # Risk of busting
                bust_prob = self._estimate_bust_probability(game_state)
                expected_value *= (1 - bust_prob)
            
            elif action == BlackjackAction.DOUBLE:
                # Double the bet, double the reward/loss
                expected_value *= 2
            
            elif action == BlackjackAction.SPLIT:
                # Two hands, more complex calculation
                expected_value *= 1.5  # Simplified
            
            elif action == BlackjackAction.SURRENDER:
                expected_value = -0.5  # Lose half the bet
            
            elif action == BlackjackAction.STAND:
                # Keep current expected value
                pass
        
        return expected_value
    
    def _estimate_bust_probability(self, game_state: BlackjackGameState) -> float:
        """Estimate probability of busting if hitting."""
        if game_state.current_hand >= len(game_state.player_hands):
            return 1.0
        
        current_hand = game_state.player_hands[game_state.current_hand]
        hand_value = current_hand.get_value()
        
        if hand_value >= 21:
            return 1.0
        
        # Calculate probability of busting based on remaining cards
        bust_cards = 0
        total_cards = sum(self.blackjack_beliefs.deck_composition_belief.values())
        
        for rank, count in self.blackjack_beliefs.deck_composition_belief.items():
            if rank >= 10:  # 10, J, Q, K
                card_value = 10
            else:
                card_value = rank
            
            if hand_value + card_value > 21:
                bust_cards += count
        
        if total_cards == 0:
            return 0.5  # Default probability
        
        return bust_cards / total_cards
    
    def compute_goal_directed_free_energy(self, action: BlackjackAction, game_state: BlackjackGameState) -> float:
        """Compute free energy with goal-directed component for blackjack."""
        # Standard expected free energy from parent class
        blackjack_action = Action(
            type=action.value,
            parameters={"game_state": game_state}
        )
        
        # This is a simplified version - in practice we'd need to implement the full conversion
        standard_efe = 0.0  # Placeholder
        
        # Goal-directed component
        expected_value = self.compute_expected_value(action, game_state)
        
        # Combine with risk tolerance
        risk_penalty = self.risk_tolerance * abs(expected_value)
        
        # Goal alignment
        goal_alignment = (
            self.goals[0] * max(0, expected_value) +  # Maximize winnings
            self.goals[1] * max(0, -expected_value) +  # Minimize losses
            self.goals[2] * 0.1 +  # Learning bonus
            self.goals[3] * (1.0 if expected_value > -0.1 else 0.0)  # Bankroll preservation
        )
        
        return standard_efe + risk_penalty - goal_alignment.item()
    
    def choose_action(self, game_state: BlackjackGameState) -> BlackjackAction:
        """Choose action using active inference principles."""
        # Update beliefs based on current state
        obs = self.game_state_to_observation(game_state)
        self.observe(obs)
        
        # Update beliefs from cards seen
        self.update_beliefs_from_cards(game_state.cards_seen)
        
        # Get current hand
        if game_state.current_hand >= len(game_state.player_hands):
            return BlackjackAction.STAND
        
        current_hand = game_state.player_hands[game_state.current_hand]
        
        # Get possible actions
        possible_actions = []
        
        if current_hand.is_blackjack():
            return BlackjackAction.STAND
        
        if current_hand.get_value() < 21:
            possible_actions.append(BlackjackAction.HIT)
        
        possible_actions.append(BlackjackAction.STAND)
        
        if current_hand.can_double():
            possible_actions.append(BlackjackAction.DOUBLE)
        
        if current_hand.can_split():
            possible_actions.append(BlackjackAction.SPLIT)
        
        # Allow surrender on first two cards
        if len(current_hand.cards) == 2:
            possible_actions.append(BlackjackAction.SURRENDER)
        
        # Compute expected free energy for each action
        action_values = {}
        for action in possible_actions:
            efe = self.compute_goal_directed_free_energy(action, game_state)
            action_values[action] = efe
        
        # Select action with minimum expected free energy
        best_action = min(action_values.keys(), key=lambda a: action_values[a])
        
        # Add some exploration based on uncertainty
        if self.beliefs.confidence < 0.5:
            # More exploratory when uncertain
            if np.random.random() < 0.1:
                return np.random.choice(possible_actions)
        
        return best_action
    
    def insurance_decision(self, game_state: BlackjackGameState) -> bool:
        """Decide whether to take insurance using active inference."""
        # Insurance is generally a bad bet unless counting cards
        if self.blackjack_beliefs.card_count_belief > 2:
            return True
        
        # Use dealer hole card beliefs
        ten_value_prob = sum(
            self.blackjack_beliefs.dealer_hole_card_belief.get(i, 0)
            for i in [10, 11, 12, 13]  # 10, J, Q, K
        )
        
        # Take insurance if probability of dealer blackjack > 1/3
        return ten_value_prob > 1/3
    
    def make_bet(self, min_bet: float, max_bet: float) -> float:
        """Make a bet using active inference and card counting."""
        # Base bet
        base_bet = min_bet
        
        # Adjust based on card count (Kelly criterion approximation)
        if self.betting_strategy == "kelly":
            true_count = self.blackjack_beliefs.card_count_belief
            if true_count > 1:
                # Increase bet with positive count
                bet_multiplier = min(true_count, 5)
                base_bet *= bet_multiplier
        
        elif self.betting_strategy == "progressive":
            # Increase bet after wins, decrease after losses
            if hasattr(self, 'last_result') and self.last_result > 0:
                base_bet *= 1.5
            elif hasattr(self, 'last_result') and self.last_result < 0:
                base_bet *= 0.75
        
        # Risk management
        max_risk_bet = self.bankroll * 0.05  # Never bet more than 5% of bankroll
        
        final_bet = min(max(base_bet, min_bet), max_bet, max_risk_bet)
        
        return final_bet
    
    def learn_from_result(self, result: float, game_state: BlackjackGameState) -> None:
        """Learn from game result using active inference."""
        # Store result for betting strategy
        self.last_result = result
        
        # Update goals based on result
        self.update_goals(result)
        
        # Create training data
        obs = self.game_state_to_observation(game_state)
        
        # Target outcome
        if result > 0:
            target_outcome = torch.tensor([1.0, 0.0, 0.0])  # Win
        elif result < 0:
            target_outcome = torch.tensor([0.0, 1.0, 0.0])  # Lose
        else:
            target_outcome = torch.tensor([0.0, 0.0, 1.0])  # Push
        
        # Target value
        target_value = torch.tensor([result])
        
        # Forward pass
        model_output = self.blackjack_model(obs.data.unsqueeze(0))
        
        # Compute losses
        outcome_loss = nn.CrossEntropyLoss()(
            model_output['outcome_probs'], 
            target_outcome.unsqueeze(0)
        )
        
        value_loss = nn.MSELoss()(
            model_output['expected_value'], 
            target_value.unsqueeze(0)
        )
        
        total_loss = outcome_loss + value_loss
        
        # Backward pass
        self.blackjack_optimizer.zero_grad()
        total_loss.backward()
        self.blackjack_optimizer.step()
        
        # Perform metacognitive reflection
        self.metacognitive_reflection()
    
    def _action_to_tensor(self, action: Action) -> torch.Tensor:
        """Convert action to tensor representation."""
        # Simple one-hot encoding for blackjack actions
        action_map = {
            "hit": 0,
            "stand": 1,
            "double": 2,
            "split": 3,
            "surrender": 4
        }
        
        action_tensor = torch.zeros(5)
        if action.type in action_map:
            action_tensor[action_map[action.type]] = 1.0
        
        return action_tensor
    
    def get_possible_actions(self) -> List[Action]:
        """Get all possible actions in blackjack."""
        return [
            Action(type="hit", parameters={}),
            Action(type="stand", parameters={}),
            Action(type="double", parameters={}),
            Action(type="split", parameters={}),
            Action(type="surrender", parameters={})
        ]
    
    def __str__(self) -> str:
        return f"ActiveInference-{self.name} (${self.bankroll:.2f}, Count: {self.blackjack_beliefs.card_count_belief:.1f})"
