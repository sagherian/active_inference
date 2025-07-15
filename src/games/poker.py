"""
Texas Hold'em Poker Game Environment for Active Inference Agents

Implements complete Texas Hold'em poker with:
- Multi-player support
- Betting rounds and actions
- Hand evaluation and showdown
- Detailed game state representation for agents
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Tuple, Set
import random
import numpy as np
from abc import ABC, abstractmethod
from itertools import combinations


class PokerAction(Enum):
    """Possible actions in poker."""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


class HandRank(Enum):
    """Poker hand rankings."""
    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10


class BettingRound(Enum):
    """Betting rounds in Texas Hold'em."""
    PRE_FLOP = "pre_flop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


@dataclass
class PokerHand:
    """Represents a poker hand with ranking."""
    cards: List['Card']
    rank: HandRank
    kickers: List[int]
    
    def __lt__(self, other: 'PokerHand') -> bool:
        """Compare poker hands."""
        if self.rank.value != other.rank.value:
            return self.rank.value < other.rank.value
        return self.kickers < other.kickers
    
    def __str__(self) -> str:
        return f"{self.rank.name}: {', '.join(str(card) for card in self.cards)}"


@dataclass
class PotInfo:
    """Information about a pot."""
    amount: float
    eligible_players: Set[str]
    is_side_pot: bool = False


@dataclass
class PokerGameState:
    """Complete representation of poker game state."""
    player_cards: Dict[str, List['Card']]
    community_cards: List['Card']
    pot: float
    side_pots: List[PotInfo]
    current_bets: Dict[str, float]
    player_stacks: Dict[str, float]
    betting_round: BettingRound
    current_player: str
    active_players: List[str]
    folded_players: Set[str]
    all_in_players: Set[str]
    dealer_position: int
    small_blind: float
    big_blind: float
    min_raise: float
    cards_seen: List['Card']
    
    def to_tensor(self) -> np.ndarray:
        """Convert game state to tensor for neural networks."""
        features = []
        
        # Player's hole cards (if available)
        if self.current_player in self.player_cards:
            hole_cards = self.player_cards[self.current_player]
            for card in hole_cards:
                features.extend([card.rank.value, card.suit.value])
        else:
            features.extend([0, 0, 0, 0])  # Unknown cards
        
        # Community cards
        for i in range(5):
            if i < len(self.community_cards):
                card = self.community_cards[i]
                features.extend([card.rank.value, card.suit.value])
            else:
                features.extend([0, 0])  # No card yet
        
        # Pot information
        features.append(self.pot)
        features.append(len(self.side_pots))
        
        # Betting information
        features.append(self.current_bets.get(self.current_player, 0))
        features.append(max(self.current_bets.values()) if self.current_bets else 0)
        features.append(self.min_raise)
        
        # Position information
        features.append(self.dealer_position)
        features.append(len(self.active_players))
        features.append(self.betting_round.value)
        
        # Stack information
        features.append(self.player_stacks.get(self.current_player, 0))
        features.append(np.mean(list(self.player_stacks.values())))
        
        return np.array(features, dtype=np.float32)


class HandEvaluator:
    """Evaluates poker hands."""
    
    @staticmethod
    def evaluate_hand(cards: List['Card']) -> PokerHand:
        """Evaluate a 5-7 card poker hand."""
        if len(cards) < 5:
            raise ValueError("Need at least 5 cards to evaluate hand")
        
        # Find the best 5-card hand
        best_hand = None
        for five_cards in combinations(cards, 5):
            hand = HandEvaluator._evaluate_five_cards(list(five_cards))
            if best_hand is None or hand > best_hand:
                best_hand = hand
        
        return best_hand
    
    @staticmethod
    def _evaluate_five_cards(cards: List['Card']) -> PokerHand:
        """Evaluate exactly 5 cards."""
        cards = sorted(cards, key=lambda c: c.rank.value, reverse=True)
        
        # Check for flush
        suits = [card.suit for card in cards]
        is_flush = len(set(suits)) == 1
        
        # Check for straight
        ranks = [card.rank.value for card in cards]
        is_straight = HandEvaluator._is_straight(ranks)
        
        # Special case: A-2-3-4-5 straight (wheel)
        if ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            ranks = [5, 4, 3, 2, 1]  # Ace low
        
        # Count rank frequencies
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        counts = sorted(rank_counts.values(), reverse=True)
        unique_ranks = sorted(rank_counts.keys(), key=lambda r: (rank_counts[r], r), reverse=True)
        
        # Determine hand rank
        if is_straight and is_flush:
            if ranks == [14, 13, 12, 11, 10]:
                return PokerHand(cards, HandRank.ROYAL_FLUSH, [14])
            else:
                return PokerHand(cards, HandRank.STRAIGHT_FLUSH, [max(ranks)])
        elif counts == [4, 1]:
            return PokerHand(cards, HandRank.FOUR_OF_A_KIND, unique_ranks)
        elif counts == [3, 2]:
            return PokerHand(cards, HandRank.FULL_HOUSE, unique_ranks)
        elif is_flush:
            return PokerHand(cards, HandRank.FLUSH, sorted(ranks, reverse=True))
        elif is_straight:
            return PokerHand(cards, HandRank.STRAIGHT, [max(ranks)])
        elif counts == [3, 1, 1]:
            return PokerHand(cards, HandRank.THREE_OF_A_KIND, unique_ranks)
        elif counts == [2, 2, 1]:
            return PokerHand(cards, HandRank.TWO_PAIR, unique_ranks)
        elif counts == [2, 1, 1, 1]:
            return PokerHand(cards, HandRank.ONE_PAIR, unique_ranks)
        else:
            return PokerHand(cards, HandRank.HIGH_CARD, sorted(ranks, reverse=True))
    
    @staticmethod
    def _is_straight(ranks: List[int]) -> bool:
        """Check if ranks form a straight."""
        sorted_ranks = sorted(set(ranks))
        if len(sorted_ranks) != 5:
            return False
        
        return sorted_ranks[-1] - sorted_ranks[0] == 4


class PokerPlayer(ABC):
    """Abstract base class for poker players."""
    
    def __init__(self, name: str, stack: float):
        self.name = name
        self.stack = stack
        self.hole_cards: List['Card'] = []
        self.current_bet = 0.0
        self.total_bet = 0.0
        self.is_folded = False
        self.is_all_in = False
        self.position = 0
    
    @abstractmethod
    def make_decision(self, game_state: PokerGameState) -> Tuple[PokerAction, float]:
        """Make a decision given the current game state."""
        pass
    
    def reset_for_new_hand(self) -> None:
        """Reset player state for a new hand."""
        self.hole_cards = []
        self.current_bet = 0.0
        self.total_bet = 0.0
        self.is_folded = False
        self.is_all_in = False
    
    def add_hole_card(self, card: 'Card') -> None:
        """Add a hole card."""
        self.hole_cards.append(card)
    
    def place_bet(self, amount: float) -> float:
        """Place a bet, returning the actual amount bet."""
        actual_amount = min(amount, self.stack)
        self.stack -= actual_amount
        self.current_bet += actual_amount
        self.total_bet += actual_amount
        
        if self.stack == 0:
            self.is_all_in = True
        
        return actual_amount
    
    def fold(self) -> None:
        """Fold the hand."""
        self.is_folded = True
    
    def __str__(self) -> str:
        return f"{self.name} (${self.stack:.2f})"


class TexasHoldemGame:
    """Complete Texas Hold'em poker game implementation."""
    
    def __init__(self, small_blind: float = 1.0, big_blind: float = 2.0):
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.players: List[PokerPlayer] = []
        self.deck = None
        self.community_cards: List['Card'] = []
        self.pot = 0.0
        self.side_pots: List[PotInfo] = []
        self.current_bets: Dict[str, float] = {}
        self.dealer_position = 0
        self.current_player_index = 0
        self.betting_round = BettingRound.PRE_FLOP
        self.min_raise = self.big_blind
        self.game_state: Optional[PokerGameState] = None
        self.hand_evaluator = HandEvaluator()
    
    def add_player(self, player: PokerPlayer) -> None:
        """Add a player to the game."""
        self.players.append(player)
        player.position = len(self.players) - 1
    
    def remove_player(self, player: PokerPlayer) -> None:
        """Remove a player from the game."""
        if player in self.players:
            self.players.remove(player)
            # Update positions
            for i, p in enumerate(self.players):
                p.position = i
    
    def get_active_players(self) -> List[PokerPlayer]:
        """Get players who are still active in the hand."""
        return [p for p in self.players if not p.is_folded and p.stack > 0]
    
    def get_players_in_pot(self) -> List[PokerPlayer]:
        """Get players who are eligible for the pot."""
        return [p for p in self.players if not p.is_folded]
    
    def start_new_hand(self) -> None:
        """Start a new hand."""
        if len(self.players) < 2:
            raise ValueError("Need at least 2 players to start a hand")
        
        # Reset game state
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0.0
        self.side_pots = []
        self.current_bets = {}
        self.betting_round = BettingRound.PRE_FLOP
        self.min_raise = self.big_blind
        
        # Reset players
        for player in self.players:
            player.reset_for_new_hand()
        
        # Deal hole cards
        for _ in range(2):
            for player in self.players:
                card = self.deck.deal_card()
                if card:
                    player.add_hole_card(card)
        
        # Post blinds
        self._post_blinds()
        
        # Set first player to act
        self.current_player_index = (self.dealer_position + 3) % len(self.players)
        
        self._update_game_state()
    
    def _post_blinds(self) -> None:
        """Post small and big blinds."""
        if len(self.players) == 2:
            # Heads up: dealer posts small blind
            sb_index = self.dealer_position
            bb_index = (self.dealer_position + 1) % len(self.players)
        else:
            sb_index = (self.dealer_position + 1) % len(self.players)
            bb_index = (self.dealer_position + 2) % len(self.players)
        
        # Post small blind
        sb_amount = self.players[sb_index].place_bet(self.small_blind)
        self.current_bets[self.players[sb_index].name] = sb_amount
        self.pot += sb_amount
        
        # Post big blind
        bb_amount = self.players[bb_index].place_bet(self.big_blind)
        self.current_bets[self.players[bb_index].name] = bb_amount
        self.pot += bb_amount
    
    def _update_game_state(self) -> None:
        """Update the game state."""
        active_players = self.get_active_players()
        
        self.game_state = PokerGameState(
            player_cards={p.name: p.hole_cards for p in self.players},
            community_cards=self.community_cards.copy(),
            pot=self.pot,
            side_pots=self.side_pots.copy(),
            current_bets=self.current_bets.copy(),
            player_stacks={p.name: p.stack for p in self.players},
            betting_round=self.betting_round,
            current_player=self.players[self.current_player_index].name if active_players else "",
            active_players=[p.name for p in active_players],
            folded_players={p.name for p in self.players if p.is_folded},
            all_in_players={p.name for p in self.players if p.is_all_in},
            dealer_position=self.dealer_position,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            min_raise=self.min_raise,
            cards_seen=self.community_cards.copy()
        )
    
    def get_valid_actions(self, player: PokerPlayer) -> List[PokerAction]:
        """Get valid actions for a player."""
        actions = []
        
        max_bet = max(self.current_bets.values()) if self.current_bets else 0
        player_bet = self.current_bets.get(player.name, 0)
        call_amount = max_bet - player_bet
        
        if call_amount == 0:
            actions.append(PokerAction.CHECK)
        else:
            actions.append(PokerAction.CALL)
        
        actions.append(PokerAction.FOLD)
        
        # Can bet or raise if player has chips
        if player.stack > 0:
            if call_amount == 0:
                actions.append(PokerAction.BET)
            else:
                actions.append(PokerAction.RAISE)
            
            actions.append(PokerAction.ALL_IN)
        
        return actions
    
    def process_action(self, player: PokerPlayer, action: PokerAction, amount: float = 0) -> bool:
        """Process a player's action. Returns True if action was valid."""
        if player.is_folded or player.is_all_in:
            return False
        
        valid_actions = self.get_valid_actions(player)
        if action not in valid_actions:
            return False
        
        max_bet = max(self.current_bets.values()) if self.current_bets else 0
        player_bet = self.current_bets.get(player.name, 0)
        call_amount = max_bet - player_bet
        
        if action == PokerAction.FOLD:
            player.fold()
        
        elif action == PokerAction.CHECK:
            if call_amount != 0:
                return False
        
        elif action == PokerAction.CALL:
            if call_amount > 0:
                actual_amount = player.place_bet(call_amount)
                self.current_bets[player.name] = player_bet + actual_amount
                self.pot += actual_amount
        
        elif action == PokerAction.BET:
            if call_amount != 0:
                return False
            if amount < self.min_raise:
                return False
            
            actual_amount = player.place_bet(amount)
            self.current_bets[player.name] = player_bet + actual_amount
            self.pot += actual_amount
            self.min_raise = amount
        
        elif action == PokerAction.RAISE:
            if call_amount == 0:
                return False
            
            total_bet = call_amount + amount
            if amount < self.min_raise:
                return False
            
            actual_amount = player.place_bet(total_bet)
            self.current_bets[player.name] = player_bet + actual_amount
            self.pot += actual_amount
            self.min_raise = amount
        
        elif action == PokerAction.ALL_IN:
            actual_amount = player.place_bet(player.stack)
            self.current_bets[player.name] = player_bet + actual_amount
            self.pot += actual_amount
        
        return True
    
    def is_betting_round_complete(self) -> bool:
        """Check if the current betting round is complete."""
        active_players = self.get_active_players()
        
        if len(active_players) <= 1:
            return True
        
        # Check if all active players have acted and bets are equal
        max_bet = max(self.current_bets.values()) if self.current_bets else 0
        
        for player in active_players:
            if not player.is_all_in:
                player_bet = self.current_bets.get(player.name, 0)
                if player_bet != max_bet:
                    return False
        
        return True
    
    def advance_to_next_round(self) -> None:
        """Advance to the next betting round."""
        # Reset current bets
        self.current_bets = {}
        
        # Deal community cards
        if self.betting_round == BettingRound.PRE_FLOP:
            # Deal flop (3 cards)
            for _ in range(3):
                card = self.deck.deal_card()
                if card:
                    self.community_cards.append(card)
            self.betting_round = BettingRound.FLOP
        
        elif self.betting_round == BettingRound.FLOP:
            # Deal turn (1 card)
            card = self.deck.deal_card()
            if card:
                self.community_cards.append(card)
            self.betting_round = BettingRound.TURN
        
        elif self.betting_round == BettingRound.TURN:
            # Deal river (1 card)
            card = self.deck.deal_card()
            if card:
                self.community_cards.append(card)
            self.betting_round = BettingRound.RIVER
        
        # Reset min raise
        self.min_raise = self.big_blind
        
        # First player to act is first active player after dealer
        active_players = self.get_active_players()
        if active_players:
            for i in range(len(self.players)):
                idx = (self.dealer_position + 1 + i) % len(self.players)
                if self.players[idx] in active_players:
                    self.current_player_index = idx
                    break
    
    def play_betting_round(self) -> None:
        """Play a complete betting round."""
        active_players = self.get_active_players()
        
        if len(active_players) <= 1:
            return
        
        players_to_act = active_players.copy()
        
        while not self.is_betting_round_complete() and len(self.get_active_players()) > 1:
            current_player = self.players[self.current_player_index]
            
            if current_player.is_folded or current_player.is_all_in:
                self._next_player()
                continue
            
            self._update_game_state()
            
            # Get player's decision
            action, amount = current_player.make_decision(self.game_state)
            
            # Process the action
            if not self.process_action(current_player, action, amount):
                # Invalid action, default to fold
                current_player.fold()
            
            self._next_player()
    
    def _next_player(self) -> None:
        """Move to the next player."""
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
    
    def showdown(self) -> Dict[str, float]:
        """Conduct showdown and return winnings."""
        active_players = self.get_players_in_pot()
        
        if not active_players:
            return {}
        
        # Evaluate hands
        hand_values = {}
        for player in active_players:
            if len(player.hole_cards) == 2:
                all_cards = player.hole_cards + self.community_cards
                hand_values[player.name] = self.hand_evaluator.evaluate_hand(all_cards)
        
        # Find winners
        if len(hand_values) == 1:
            # Only one player left
            winner = active_players[0]
            winnings = {winner.name: self.pot}
            winner.stack += self.pot
            return winnings
        
        # Multiple players - compare hands
        sorted_hands = sorted(hand_values.items(), key=lambda x: x[1], reverse=True)
        
        # Handle ties
        winners = []
        best_hand = sorted_hands[0][1]
        
        for player_name, hand in sorted_hands:
            if hand == best_hand:
                winners.append(player_name)
            else:
                break
        
        # Distribute pot
        winnings = {}
        pot_share = self.pot / len(winners)
        
        for winner_name in winners:
            winnings[winner_name] = pot_share
            winner = next(p for p in self.players if p.name == winner_name)
            winner.stack += pot_share
        
        return winnings
    
    def play_hand(self) -> Dict[str, float]:
        """Play a complete hand of poker."""
        self.start_new_hand()
        
        # Pre-flop betting
        self.play_betting_round()
        
        # Flop, turn, river
        for _ in range(3):
            if len(self.get_active_players()) > 1:
                self.advance_to_next_round()
                self.play_betting_round()
        
        # Showdown
        winnings = self.showdown()
        
        # Move dealer button
        self.dealer_position = (self.dealer_position + 1) % len(self.players)
        
        return winnings
    
    def __str__(self) -> str:
        """String representation of the game."""
        result = f"Texas Hold'em - {len(self.players)} players\n"
        result += f"Community: {', '.join(str(card) for card in self.community_cards)}\n"
        result += f"Pot: ${self.pot:.2f}\n"
        result += f"Betting Round: {self.betting_round.value}\n"
        
        for player in self.players:
            status = ""
            if player.is_folded:
                status = " (FOLDED)"
            elif player.is_all_in:
                status = " (ALL-IN)"
            
            result += f"{player.name}: ${player.stack:.2f}{status}\n"
        
        return result


# Import the Card class from blackjack (reusing the same card implementation)
from .blackjack import Card, Deck
