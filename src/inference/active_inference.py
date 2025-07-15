"""
Core Active Inference Framework

Implements Friston's active inference principles combined with Michael Levin's
concepts of agency and goal-directed behavior for game-playing agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import numpy as np
from scipy import stats


@dataclass
class BeliefState:
    """
    Represents an agent's beliefs about the world state.
    
    In active inference, beliefs are probabilistic representations that
    the agent maintains about hidden states in the environment.
    """
    mean: torch.Tensor
    covariance: torch.Tensor
    confidence: float
    timestamp: float


@dataclass
class Observation:
    """
    Represents sensory observations received by the agent.
    
    These are the only direct access the agent has to the world,
    everything else must be inferred.
    """
    data: torch.Tensor
    timestamp: float
    source: str


@dataclass
class Action:
    """
    Represents an action the agent can take.
    
    In active inference, actions are selected to minimize expected free energy.
    """
    type: str
    parameters: Dict[str, Any]
    expected_outcome: Optional[torch.Tensor] = None
    confidence: float = 0.0


class GenerativeModel(nn.Module):
    """
    The agent's generative model of how observations are generated.
    
    This is central to active inference - the agent uses this model to:
    1. Predict future observations
    2. Infer hidden states
    3. Select actions
    """
    
    def __init__(self, observation_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # Encoder: observations -> hidden states
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder: hidden states -> observations
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, observation_dim),
            nn.Sigmoid()
        )
        
        # Dynamics model: hidden states + actions -> next hidden states
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Prior network: learns prior beliefs
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)  # mean and log_var
        )
    
    def encode(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observations into hidden state distribution."""
        h = self.encoder(observations)
        return h, torch.ones_like(h) * 0.1  # Simple covariance for now
    
    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Decode hidden states into predicted observations."""
        return self.decoder(hidden_states)
    
    def predict_next_state(self, hidden_states: torch.Tensor, 
                          actions: torch.Tensor) -> torch.Tensor:
        """Predict next hidden state given current state and action."""
        combined = torch.cat([hidden_states, actions], dim=-1)
        return self.dynamics(combined)
    
    def sample_prior(self, batch_size: int) -> torch.Tensor:
        """Sample from prior beliefs."""
        return torch.randn(batch_size, self.hidden_dim)


class FreeEnergyCalculator:
    """
    Calculates free energy for active inference.
    
    Free energy = Accuracy (negative log-likelihood) + Complexity (KL divergence)
    """
    
    @staticmethod
    def compute_accuracy(observations: torch.Tensor, 
                        predictions: torch.Tensor) -> torch.Tensor:
        """Compute accuracy term (negative log-likelihood)."""
        return -torch.distributions.Normal(predictions, 0.1).log_prob(observations).sum()
    
    @staticmethod
    def compute_complexity(posterior_mean: torch.Tensor,
                          posterior_var: torch.Tensor,
                          prior_mean: torch.Tensor,
                          prior_var: torch.Tensor) -> torch.Tensor:
        """Compute complexity term (KL divergence between posterior and prior)."""
        posterior = torch.distributions.Normal(posterior_mean, posterior_var)
        prior = torch.distributions.Normal(prior_mean, prior_var)
        return torch.distributions.kl_divergence(posterior, prior).sum()
    
    @classmethod
    def compute_free_energy(cls, observations: torch.Tensor,
                           predictions: torch.Tensor,
                           posterior_mean: torch.Tensor,
                           posterior_var: torch.Tensor,
                           prior_mean: torch.Tensor,
                           prior_var: torch.Tensor) -> torch.Tensor:
        """Compute total free energy."""
        accuracy = cls.compute_accuracy(observations, predictions)
        complexity = cls.compute_complexity(posterior_mean, posterior_var,
                                          prior_mean, prior_var)
        return accuracy + complexity


class ActiveInferenceAgent(ABC):
    """
    Base class for active inference agents.
    
    Implements core active inference loop:
    1. Observe environment
    2. Update beliefs (minimize free energy)
    3. Select action (minimize expected free energy)
    4. Act in environment
    """
    
    def __init__(self, observation_dim: int, hidden_dim: int, action_dim: int):
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # Initialize models
        self.generative_model = GenerativeModel(observation_dim, hidden_dim, action_dim)
        self.free_energy_calc = FreeEnergyCalculator()
        
        # Initialize beliefs
        self.beliefs = BeliefState(
            mean=torch.zeros(hidden_dim),
            covariance=torch.eye(hidden_dim),
            confidence=0.0,
            timestamp=0.0
        )
        
        # Hyperparameters
        self.learning_rate = 0.01
        self.precision = 1.0  # Precision of observations
        self.temperature = 1.0  # Temperature for action selection
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.generative_model.parameters(), 
                                        lr=self.learning_rate)
    
    def observe(self, observation: Observation) -> None:
        """
        Process a new observation and update beliefs.
        
        This implements the perception part of active inference where
        the agent updates its beliefs about hidden states.
        """
        # Encode observation
        encoded_mean, encoded_var = self.generative_model.encode(observation.data)
        
        # Update beliefs using Bayesian inference
        self._update_beliefs(encoded_mean, encoded_var)
    
    def _update_beliefs(self, observation_mean: torch.Tensor, 
                       observation_var: torch.Tensor) -> None:
        """Update beliefs using Bayesian inference."""
        # Prior beliefs
        prior_mean = self.beliefs.mean
        prior_var = torch.diag(self.beliefs.covariance)
        
        # Posterior beliefs (Bayesian update)
        posterior_precision = 1.0 / prior_var + 1.0 / observation_var
        posterior_var = 1.0 / posterior_precision
        posterior_mean = posterior_var * (prior_mean / prior_var + 
                                        observation_mean / observation_var)
        
        # Update belief state
        self.beliefs.mean = posterior_mean
        self.beliefs.covariance = torch.diag(posterior_var)
        self.beliefs.confidence = torch.mean(1.0 / posterior_var).item()
    
    def predict_next_observation(self, action: Action) -> torch.Tensor:
        """Predict next observation given current beliefs and action."""
        # Convert action to tensor
        action_tensor = self._action_to_tensor(action)
        
        # Predict next hidden state
        next_hidden = self.generative_model.predict_next_state(
            self.beliefs.mean.unsqueeze(0), action_tensor.unsqueeze(0)
        )
        
        # Decode to observation
        predicted_obs = self.generative_model.decode(next_hidden)
        return predicted_obs.squeeze(0)
    
    def compute_expected_free_energy(self, action: Action) -> float:
        """
        Compute expected free energy for a given action.
        
        This is used for action selection - agents choose actions that
        minimize expected free energy.
        """
        # Predict outcome of action
        predicted_obs = self.predict_next_observation(action)
        
        # Compute expected accuracy (how well we can predict)
        expected_accuracy = self._compute_expected_accuracy(predicted_obs)
        
        # Compute expected complexity (information gain)
        expected_complexity = self._compute_expected_complexity(action)
        
        return expected_accuracy + expected_complexity
    
    def _compute_expected_accuracy(self, predicted_obs: torch.Tensor) -> float:
        """Compute expected accuracy term."""
        # This is a simplified version - in practice this would involve
        # integrating over possible future observations
        return torch.mean(predicted_obs ** 2).item()
    
    def _compute_expected_complexity(self, action: Action) -> float:
        """Compute expected complexity term."""
        # This represents the information gain from the action
        # Higher complexity = more informative action
        return 0.1  # Simplified for now
    
    def select_action(self, possible_actions: List[Action]) -> Action:
        """
        Select action that minimizes expected free energy.
        
        This implements the action selection part of active inference.
        """
        if not possible_actions:
            raise ValueError("No possible actions provided")
        
        # Compute expected free energy for each action
        expected_free_energies = []
        for action in possible_actions:
            efe = self.compute_expected_free_energy(action)
            expected_free_energies.append(efe)
        
        # Select action with minimum expected free energy
        # Add some stochasticity via softmax
        probs = torch.softmax(-torch.tensor(expected_free_energies) / self.temperature, dim=0)
        selected_idx = torch.multinomial(probs, 1).item()
        
        return possible_actions[selected_idx]
    
    def learn_from_experience(self, observation: Observation, 
                             action: Action, next_observation: Observation) -> None:
        """
        Learn from experience by updating the generative model.
        
        This implements the learning aspect of active inference.
        """
        # Convert to tensors
        obs_tensor = observation.data
        action_tensor = self._action_to_tensor(action)
        next_obs_tensor = next_observation.data
        
        # Forward pass
        hidden_mean, hidden_var = self.generative_model.encode(obs_tensor)
        predicted_obs = self.generative_model.decode(hidden_mean)
        next_hidden = self.generative_model.predict_next_state(hidden_mean, action_tensor)
        predicted_next_obs = self.generative_model.decode(next_hidden)
        
        # Compute loss (free energy)
        reconstruction_loss = nn.MSELoss()(predicted_obs, obs_tensor)
        prediction_loss = nn.MSELoss()(predicted_next_obs, next_obs_tensor)
        
        # Prior loss (regularization)
        prior_loss = torch.mean(hidden_mean ** 2)
        
        total_loss = reconstruction_loss + prediction_loss + 0.1 * prior_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
    @abstractmethod
    def _action_to_tensor(self, action: Action) -> torch.Tensor:
        """Convert action to tensor representation."""
        pass
    
    @abstractmethod
    def get_possible_actions(self) -> List[Action]:
        """Get all possible actions in current state."""
        pass


class LevinianAgent(ActiveInferenceAgent):
    """
    Extension of ActiveInferenceAgent incorporating Michael Levin's ideas
    about agency, goal-directed behavior, and intelligence.
    
    Key additions:
    - Goal representations and goal-directed behavior
    - Metacognitive abilities (thinking about thinking)
    - Adaptive goal formation
    - Collective intelligence concepts
    """
    
    def __init__(self, observation_dim: int, hidden_dim: int, action_dim: int):
        super().__init__(observation_dim, hidden_dim, action_dim)
        
        # Goal representation
        self.goals = torch.zeros(hidden_dim)
        self.goal_weights = torch.ones(hidden_dim)
        
        # Metacognitive components
        self.meta_beliefs = BeliefState(
            mean=torch.zeros(hidden_dim),
            covariance=torch.eye(hidden_dim),
            confidence=0.0,
            timestamp=0.0
        )
        
        # Adaptive goal formation
        self.goal_formation_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
    
    def update_goals(self, reward: float) -> None:
        """Update goals based on experience (Levinian adaptive goal formation)."""
        # Combine current beliefs and meta-beliefs
        combined_state = torch.cat([self.beliefs.mean, self.meta_beliefs.mean])
        
        # Generate new goals
        new_goals = self.goal_formation_net(combined_state)
        
        # Update goals with moving average
        self.goals = 0.9 * self.goals + 0.1 * new_goals
        
        # Update goal weights based on reward
        if reward > 0:
            self.goal_weights *= 1.1
        else:
            self.goal_weights *= 0.9
        
        # Normalize weights
        self.goal_weights = torch.clamp(self.goal_weights, 0.1, 2.0)
    
    def compute_goal_directed_free_energy(self, action: Action) -> float:
        """
        Compute free energy with goal-directed component.
        
        This incorporates Levin's ideas about goal-directed behavior
        into the free energy calculation.
        """
        # Standard expected free energy
        standard_efe = self.compute_expected_free_energy(action)
        
        # Goal-directed component
        predicted_state = self.predict_next_observation(action)
        goal_alignment = torch.dot(predicted_state, self.goals * self.goal_weights)
        
        # Combine standard free energy with goal alignment
        return standard_efe - goal_alignment.item()
    
    def select_action(self, possible_actions: List[Action]) -> Action:
        """Select action using goal-directed free energy."""
        if not possible_actions:
            raise ValueError("No possible actions provided")
        
        # Compute goal-directed expected free energy for each action
        expected_free_energies = []
        for action in possible_actions:
            efe = self.compute_goal_directed_free_energy(action)
            expected_free_energies.append(efe)
        
        # Select action with minimum expected free energy
        probs = torch.softmax(-torch.tensor(expected_free_energies) / self.temperature, dim=0)
        selected_idx = torch.multinomial(probs, 1).item()
        
        return possible_actions[selected_idx]
    
    def metacognitive_reflection(self) -> None:
        """
        Perform metacognitive reflection on beliefs and goals.
        
        This implements Levin's ideas about metacognition and
        higher-order thinking.
        """
        # Update meta-beliefs about our own beliefs
        belief_uncertainty = torch.trace(self.beliefs.covariance)
        goal_coherence = torch.mean(self.goal_weights)
        
        # Simple metacognitive update
        self.meta_beliefs.mean = 0.9 * self.meta_beliefs.mean + 0.1 * self.beliefs.mean
        self.meta_beliefs.confidence = 0.9 * self.meta_beliefs.confidence + 0.1 * (1.0 / belief_uncertainty)
        
        # Adjust temperature based on metacognitive assessment
        if self.meta_beliefs.confidence > 0.8:
            self.temperature = 0.1  # More deterministic when confident
        else:
            self.temperature = 1.0  # More exploratory when uncertain
