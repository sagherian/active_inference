"""
Blackjack Game Environment for Active Inference Agents

Implements a complete blackjack game with support for:
- Multiple players (agents and humans)
- Standard blackjack rules
- Card counting opportunities
- Detailed game state representation
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Tuple
import random
import numpy as np
from abc import ABC, abstractmethod


class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"


class Rank(Enum):
    TWO = (2, "2")
    THREE = (3, "3")
    FOUR = (4, "4")
    FIVE = (5, "5")
    SIX = (6, "6")
    SEVEN = (7, "7")
    EIGHT = (8, "8")
    NINE = (9, "9")
    TEN = (10, "10")
    JACK = (10, "J")
    QUEEN = (10, "Q")
    KING = (10, "K")
    ACE = (11, "A")  # Can be 1 or 11
    
    def __init__(self, value: int, symbol: str):
        self.value = value
        self.symbol = symbol


@dataclass
class Card:
    """Represents a playing card."""
    suit: Suit
    rank: Rank
    
    def __str__(self) -> str:
        return f"{self.rank.symbol}{self.suit.value}"
    
    def __repr__(self) -> str:
        return self.__str__()


class Deck:
    """Standard 52-card deck with shuffling capabilities."""
    
    def __init__(self, num_decks: int = 1):
        self.num_decks = num_decks
        self.cards: List[Card] = []
        self.reset()
    
    def reset(self) -> None:
        """Reset and shuffle the deck."""
        self.cards = []
        for _ in range(self.num_decks):
            for suit in Suit:
                for rank in Rank:
                    self.cards.append(Card(suit, rank))
        self.shuffle()
    
    def shuffle(self) -> None:
        """Shuffle the deck."""
        random.shuffle(self.cards)
    
    def deal_card(self) -> Optional[Card]:
        """Deal one card from the deck."""
        if self.cards:
            return self.cards.pop()
        return None
    
    def cards_remaining(self) -> int:
        """Number of cards remaining in deck."""
        return len(self.cards)
    
    def penetration(self) -> float:
        """Deck penetration (fraction of cards dealt)."""
        total_cards = 52 * self.num_decks
        return 1.0 - (len(self.cards) / total_cards)


class Hand:
    """Represents a blackjack hand."""
    
    def __init__(self):
        self.cards: List[Card] = []
        self.bet: float = 0.0
        self.is_split: bool = False
        self.is_doubled: bool = False
        self.is_surrendered: bool = False
    
    def add_card(self, card: Card) -> None:
        """Add a card to the hand."""
        self.cards.append(card)
    
    def get_value(self) -> int:
        """Calculate hand value with optimal ace handling."""
        total = 0
        aces = 0
        
        for card in self.cards:
            if card.rank == Rank.ACE:
                aces += 1
                total += 11
            else:
                total += card.rank.value
        
        # Adjust for aces
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        
        return total
    
    def is_blackjack(self) -> bool:
        """Check if hand is blackjack (21 with 2 cards)."""
        return len(self.cards) == 2 and self.get_value() == 21
    
    def is_bust(self) -> bool:
        """Check if hand is busted (over 21)."""
        return self.get_value() > 21
    
    def is_soft(self) -> bool:
        """Check if hand is soft (contains an ace counted as 11)."""
        value = 0
        has_ace = False
        
        for card in self.cards:
            if card.rank == Rank.ACE:
                has_ace = True
            value += card.rank.value
        
        return has_ace and value <= 21 and (value - 10) >= 0
    
    def can_split(self) -> bool:
        """Check if hand can be split."""
        return (len(self.cards) == 2 and 
                self.cards[0].rank == self.cards[1].rank and
                not self.is_split)
    
    def can_double(self) -> bool:
        """Check if hand can be doubled."""
        return len(self.cards) == 2 and not self.is_doubled
    
    def __str__(self) -> str:
        cards_str = ", ".join(str(card) for card in self.cards)
        return f"[{cards_str}] = {self.get_value()}"


class BlackjackAction(Enum):
    """Possible actions in blackjack."""
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"
    SURRENDER = "surrender"
    INSURANCE = "insurance"


@dataclass
class BlackjackGameState:
    """Complete representation of blackjack game state."""
    player_hands: List[Hand]
    dealer_hand: Hand
    deck: Deck
    current_player: int
    current_hand: int
    phase: str  # "betting", "playing", "dealer", "finished"
    insurance_available: bool
    cards_seen: List[Card]
    running_count: int
    true_count: float
    
    def to_tensor(self) -> np.ndarray:
        """Convert game state to tensor for neural networks."""
        # This is a simplified representation
        # In practice, you'd want more sophisticated encoding
        features = []
        
        # Current hand value
        if self.current_hand < len(self.player_hands):
            features.append(self.player_hands[self.current_hand].get_value())
        else:
            features.append(0)
        
        # Dealer up card
        if self.dealer_hand.cards:
            features.append(self.dealer_hand.cards[0].rank.value)
        else:
            features.append(0)
        
        # Deck penetration
        features.append(self.deck.penetration())
        
        # True count (for card counting)
        features.append(self.true_count)
        
        # Hand characteristics
        current_hand = self.player_hands[self.current_hand] if self.current_hand < len(self.player_hands) else Hand()
        features.extend([
            1.0 if current_hand.is_soft() else 0.0,
            1.0 if current_hand.can_split() else 0.0,
            1.0 if current_hand.can_double() else 0.0,
            1.0 if current_hand.is_blackjack() else 0.0
        ])
        
        return np.array(features, dtype=np.float32)


class BlackjackPlayer(ABC):
    """Abstract base class for blackjack players."""
    
    def __init__(self, name: str, bankroll: float):
        self.name = name
        self.bankroll = bankroll
        self.hands: List[Hand] = []
        self.insurance_bet: float = 0.0
    
    @abstractmethod
    def make_bet(self, min_bet: float, max_bet: float) -> float:
        """Make a bet for the next hand."""
        pass
    
    @abstractmethod
    def choose_action(self, game_state: BlackjackGameState) -> BlackjackAction:
        """Choose an action given the current game state."""
        pass
    
    @abstractmethod
    def insurance_decision(self, game_state: BlackjackGameState) -> bool:
        """Decide whether to take insurance."""
        pass
    
    def add_hand(self, hand: Hand) -> None:
        """Add a hand to the player."""
        self.hands.append(hand)
    
    def clear_hands(self) -> None:
        """Clear all hands."""
        self.hands = []
    
    def adjust_bankroll(self, amount: float) -> None:
        """Adjust bankroll by amount (positive = win, negative = loss)."""
        self.bankroll += amount


class BlackjackGame:
    """Complete blackjack game implementation."""
    
    def __init__(self, num_decks: int = 6, min_bet: float = 5.0, max_bet: float = 1000.0):
        self.deck = Deck(num_decks)
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.players: List[BlackjackPlayer] = []
        self.dealer_hand = Hand()
        self.cards_seen: List[Card] = []
        self.running_count = 0
        self.game_state: Optional[BlackjackGameState] = None
        
        # Card counting system (Hi-Lo)
        self.card_values = {
            Rank.TWO: 1, Rank.THREE: 1, Rank.FOUR: 1, Rank.FIVE: 1, Rank.SIX: 1,
            Rank.SEVEN: 0, Rank.EIGHT: 0, Rank.NINE: 0,
            Rank.TEN: -1, Rank.JACK: -1, Rank.QUEEN: -1, Rank.KING: -1, Rank.ACE: -1
        }
    
    def add_player(self, player: BlackjackPlayer) -> None:
        """Add a player to the game."""
        self.players.append(player)
    
    def remove_player(self, player: BlackjackPlayer) -> None:
        """Remove a player from the game."""
        if player in self.players:
            self.players.remove(player)
    
    def deal_card(self) -> Optional[Card]:
        """Deal a card and update card counting."""
        card = self.deck.deal_card()
        if card:
            self.cards_seen.append(card)
            self.running_count += self.card_values[card.rank]
        return card
    
    def calculate_true_count(self) -> float:
        """Calculate true count for card counting."""
        decks_remaining = self.deck.cards_remaining() / 52
        if decks_remaining > 0:
            return self.running_count / decks_remaining
        return 0.0
    
    def update_game_state(self, phase: str, current_player: int = 0, current_hand: int = 0) -> None:
        """Update the game state."""
        self.game_state = BlackjackGameState(
            player_hands=[hand for player in self.players for hand in player.hands],
            dealer_hand=self.dealer_hand,
            deck=self.deck,
            current_player=current_player,
            current_hand=current_hand,
            phase=phase,
            insurance_available=self.is_insurance_available(),
            cards_seen=self.cards_seen.copy(),
            running_count=self.running_count,
            true_count=self.calculate_true_count()
        )
    
    def is_insurance_available(self) -> bool:
        """Check if insurance is available."""
        return (len(self.dealer_hand.cards) == 1 and 
                self.dealer_hand.cards[0].rank == Rank.ACE)
    
    def deal_initial_cards(self) -> None:
        """Deal initial cards to all players and dealer."""
        # Deal two cards to each player
        for _ in range(2):
            for player in self.players:
                if player.hands:
                    card = self.deal_card()
                    if card:
                        player.hands[0].add_card(card)
        
        # Deal one card to dealer (face up)
        card = self.deal_card()
        if card:
            self.dealer_hand.add_card(card)
    
    def play_hand(self, player: BlackjackPlayer, hand: Hand) -> None:
        """Play a single hand for a player."""
        while not hand.is_bust() and not hand.is_blackjack():
            self.update_game_state("playing", self.players.index(player), 
                                 player.hands.index(hand))
            
            action = player.choose_action(self.game_state)
            
            if action == BlackjackAction.HIT:
                card = self.deal_card()
                if card:
                    hand.add_card(card)
            
            elif action == BlackjackAction.STAND:
                break
            
            elif action == BlackjackAction.DOUBLE:
                if hand.can_double():
                    hand.is_doubled = True
                    hand.bet *= 2
                    card = self.deal_card()
                    if card:
                        hand.add_card(card)
                    break
            
            elif action == BlackjackAction.SPLIT:
                if hand.can_split():
                    # Create new hand with second card
                    new_hand = Hand()
                    new_hand.add_card(hand.cards.pop())
                    new_hand.bet = hand.bet
                    new_hand.is_split = True
                    hand.is_split = True
                    
                    # Deal new cards to both hands
                    card1 = self.deal_card()
                    card2 = self.deal_card()
                    if card1 and card2:
                        hand.add_card(card1)
                        new_hand.add_card(card2)
                    
                    player.add_hand(new_hand)
            
            elif action == BlackjackAction.SURRENDER:
                hand.is_surrendered = True
                break
    
    def play_dealer_hand(self) -> None:
        """Play the dealer's hand according to standard rules."""
        # Deal dealer's hole card
        card = self.deal_card()
        if card:
            self.dealer_hand.add_card(card)
        
        # Dealer must hit on soft 17
        while self.dealer_hand.get_value() < 17 or (
            self.dealer_hand.get_value() == 17 and self.dealer_hand.is_soft()
        ):
            card = self.deal_card()
            if card:
                self.dealer_hand.add_card(card)
    
    def determine_winner(self, player_hand: Hand) -> float:
        """Determine the result of a hand and return the payout multiplier."""
        if player_hand.is_surrendered:
            return -0.5
        
        if player_hand.is_bust():
            return -1.0
        
        if player_hand.is_blackjack() and not self.dealer_hand.is_blackjack():
            return 1.5
        
        if self.dealer_hand.is_bust():
            return 1.0
        
        player_value = player_hand.get_value()
        dealer_value = self.dealer_hand.get_value()
        
        if player_value > dealer_value:
            return 1.0
        elif player_value < dealer_value:
            return -1.0
        else:
            return 0.0  # Push
    
    def play_round(self) -> Dict[str, float]:
        """Play a complete round of blackjack."""
        if not self.players:
            return {}
        
        # Clear previous hands
        for player in self.players:
            player.clear_hands()
        self.dealer_hand = Hand()
        
        # Betting phase
        self.update_game_state("betting")
        for player in self.players:
            bet = player.make_bet(self.min_bet, self.max_bet)
            hand = Hand()
            hand.bet = min(max(bet, self.min_bet), self.max_bet)
            player.add_hand(hand)
        
        # Deal initial cards
        self.deal_initial_cards()
        
        # Insurance phase
        if self.is_insurance_available():
            for player in self.players:
                if player.insurance_decision(self.game_state):
                    player.insurance_bet = player.hands[0].bet * 0.5
        
        # Player turns
        for player in self.players:
            for hand in player.hands:
                self.play_hand(player, hand)
        
        # Dealer turn
        self.update_game_state("dealer")
        self.play_dealer_hand()
        
        # Determine winners and payouts
        results = {}
        for player in self.players:
            total_payout = 0.0
            
            for hand in player.hands:
                payout_multiplier = self.determine_winner(hand)
                payout = hand.bet * payout_multiplier
                total_payout += payout
                player.adjust_bankroll(payout)
            
            # Insurance payout
            if player.insurance_bet > 0:
                if self.dealer_hand.is_blackjack():
                    insurance_payout = player.insurance_bet * 2
                    player.adjust_bankroll(insurance_payout)
                    total_payout += insurance_payout
                else:
                    player.adjust_bankroll(-player.insurance_bet)
                    total_payout -= player.insurance_bet
                player.insurance_bet = 0.0
            
            results[player.name] = total_payout
        
        self.update_game_state("finished")
        return results
    
    def get_basic_strategy_action(self, player_hand: Hand) -> BlackjackAction:
        """Get basic strategy action for a given hand."""
        player_value = player_hand.get_value()
        dealer_up_card = self.dealer_hand.cards[0].rank.value
        
        # Simplified basic strategy
        if player_hand.can_split():
            pair_value = player_hand.cards[0].rank.value
            if pair_value in [8, 11]:  # Always split 8s and Aces
                return BlackjackAction.SPLIT
            elif pair_value == 10:  # Never split 10s
                return BlackjackAction.STAND
        
        if player_hand.is_soft():
            # Soft hand strategy
            if player_value <= 17:
                return BlackjackAction.HIT
            elif player_value == 18:
                return BlackjackAction.HIT if dealer_up_card >= 9 else BlackjackAction.STAND
            else:
                return BlackjackAction.STAND
        else:
            # Hard hand strategy
            if player_value <= 11:
                return BlackjackAction.HIT
            elif player_value == 12:
                return BlackjackAction.HIT if dealer_up_card <= 3 or dealer_up_card >= 7 else BlackjackAction.STAND
            elif 13 <= player_value <= 16:
                return BlackjackAction.HIT if dealer_up_card >= 7 else BlackjackAction.STAND
            else:
                return BlackjackAction.STAND
    
    def __str__(self) -> str:
        """String representation of the game."""
        result = f"Blackjack Game - {len(self.players)} players\n"
        result += f"Dealer: {self.dealer_hand}\n"
        for i, player in enumerate(self.players):
            result += f"Player {i+1} ({player.name}): "
            result += ", ".join(str(hand) for hand in player.hands)
            result += f" (${player.bankroll:.2f})\n"
        result += f"True Count: {self.calculate_true_count():.2f}\n"
        result += f"Deck Penetration: {self.deck.penetration():.2%}\n"
        return result
