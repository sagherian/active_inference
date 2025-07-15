# Active Inference Games

A Python-based gaming system implementing Friston's active inference framework combined with Michael Levin's concepts of intelligence and agency. Agents can play blackjack and Texas Hold'em poker against each other and humans.

## 🎯 Project Overview

This project creates intelligent agents that use active inference principles to play card games. The agents maintain beliefs about the game state, other players, and optimal strategies while minimizing free energy through predictive processing.

## 🧠 Active Inference Framework

### Core Concepts
- **Free Energy Minimization**: Agents minimize surprise through accurate predictions
- **Predictive Processing**: Continuous updating of beliefs based on observations
- **Goal-Directed Behavior**: Michael Levin's agency concepts for strategic decision-making
- **Bayesian Inference**: Probabilistic reasoning for uncertain environments

### Implementation
- **Generative Models**: Neural networks that predict game outcomes
- **Belief Updates**: Bayesian inference for state estimation
- **Action Selection**: Expected free energy minimization
- **Learning**: Continuous adaptation of model parameters

## 🎮 Games

### Blackjack
- Card counting and probability estimation
- Risk assessment and optimal stopping
- Dealer behavior modeling
- Bankroll management

### Texas Hold'em Poker
- Opponent modeling and psychology
- Bluffing and deception detection
- Betting strategy optimization
- Hand strength evaluation

## 🤖 Agent Architecture

### Multi-Agent System
- **Agent vs Agent**: Competitive gameplay between AI agents
- **Agent vs Human**: Interactive gameplay with human players
- **Learning**: Agents adapt strategies through experience
- **Communication**: Implicit coordination through game theory

### Neural Network Components
- **Perception**: State representation and feature extraction
- **Prediction**: Generative models for future states
- **Action**: Policy networks for decision-making
- **Value**: Utility estimation for different outcomes

## 🛠️ Technical Stack

- **Python 3.9+**: Core language
- **PyTorch**: Neural networks and deep learning
- **NumPy**: Numerical computations
- **SciPy**: Statistical functions
- **Matplotlib**: Visualization and analysis
- **Type Hints**: Full type safety

## 📁 Project Structure

```
active_inf_games/
├── src/
│   ├── agents/              # Active inference agents
│   ├── games/               # Game environments
│   ├── models/              # Neural network models
│   ├── inference/           # Active inference core
│   └── utils/               # Utility functions
├── tests/                   # Unit tests
├── examples/                # Usage examples
├── docs/                    # Documentation
└── requirements.txt         # Dependencies
```

## 🚀 Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Blackjack Demo**
   ```bash
   python examples/blackjack_demo.py
   ```

3. **Run Poker Tournament**
   ```bash
   python examples/poker_tournament.py
   ```

4. **Human vs Agent**
   ```bash
   python examples/human_vs_agent.py
   ```

## 📊 Features

- **Real-time Learning**: Agents adapt during gameplay
- **Visualization**: Game state and agent belief visualization
- **Analytics**: Performance metrics and strategy analysis
- **Extensible**: Easy to add new games and agent types
- **Interactive**: Human-friendly interfaces for gameplay

## 🧪 Research Applications

- **Cognitive Science**: Understanding decision-making processes
- **AI Research**: Testing active inference in complex environments
- **Game Theory**: Exploring optimal strategies and equilibria
- **Neuroscience**: Modeling predictive processing in the brain

## 📚 References

- Karl Friston's Active Inference framework
- Michael Levin's work on agency and intelligence
- Predictive processing literature
- Game theory and optimal strategy research

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines and ensure all code follows the active inference principles outlined in the project.

## 📜 License

MIT License - see LICENSE file for details
