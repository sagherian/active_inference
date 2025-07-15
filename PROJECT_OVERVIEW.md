# Active Inference Games Project Overview

## 🎯 Project Summary

The **Active Inference Games** project implements sophisticated AI agents that play card games using Karl Friston's Active Inference framework combined with Michael Levin's theories of goal-directed behavior. The system currently supports Blackjack and Texas Hold'em Poker, with agents capable of playing against each other and human players.

## 🧠 Theoretical Foundation

### Active Inference (Karl Friston)
- **Free Energy Principle**: Agents minimize prediction errors by updating beliefs and taking actions
- **Predictive Processing**: Continuous prediction and belief updating based on sensory input
- **Bayesian Brain**: Probabilistic reasoning under uncertainty
- **Hierarchical Inference**: Multi-level belief structures for complex decision-making

### Levin's Intelligence Framework (Michael Levin)
- **Goal-Directed Behavior**: Agents pursue objectives through competent navigation of problem spaces
- **Metacognitive Reflection**: Self-awareness and strategy adaptation
- **Embodied Cognition**: Intelligence emerges from agent-environment interactions
- **Developmental Intelligence**: Learning and adaptation over time

## 🏗️ Architecture

### Core Components

```
src/
├── core/
│   └── active_inference.py     # Core active inference framework
├── games/
│   ├── blackjack.py           # Blackjack game implementation
│   └── poker.py               # Texas Hold'em poker implementation
└── agents/
    ├── blackjack_agent.py     # Blackjack AI agent
    └── poker_agent.py         # Poker AI agent (to be implemented)
```

### Key Classes

1. **ActiveInferenceAgent**: Base class implementing Friston's active inference
2. **LevinianAgent**: Extension incorporating Levin's goal-directed behavior
3. **BlackjackActiveInferenceAgent**: Specialized agent for blackjack
4. **BlackjackGame**: Complete blackjack environment with multi-player support
5. **TexasHoldemGame**: Poker environment with betting mechanics

## 🎮 Game Implementations

### Blackjack Features
- Multi-deck support with configurable parameters
- Full blackjack rules (hit, stand, double, split, surrender, insurance)
- Card counting and probability tracking
- Betting strategies with bankroll management
- Basic strategy implementation for comparison

### Texas Hold'em Features
- Tournament-style play with blind structure
- Complete betting rounds (pre-flop, flop, turn, river)
- Hand evaluation and ranking
- Multi-player support with position tracking
- Pot odds calculations

## 🤖 AI Agent Capabilities

### Blackjack Agent
- **Belief Updating**: Tracks cards played and adjusts probabilities
- **Card Counting**: Sophisticated counting system for deck composition
- **Risk Assessment**: Balances potential gains against loss probability
- **Strategy Learning**: Adapts play based on historical outcomes
- **Bankroll Management**: Dynamic betting based on confidence and risk tolerance

### Poker Agent (Planned)
- **Opponent Modeling**: Builds behavioral models of other players
- **Bluffing Detection**: Identifies deceptive play patterns
- **Position Awareness**: Adjusts strategy based on betting position
- **Pot Odds Calculation**: Mathematical decision-making for betting
- **Tournament Strategy**: Adapts to changing blind structures

## 🔬 Technical Implementation

### Neural Networks
- **PyTorch**: Deep learning framework for generative models
- **Variational Inference**: Approximate Bayesian inference
- **Attention Mechanisms**: Focus on relevant game states
- **Reinforcement Learning**: Q-learning for strategy optimization

### Mathematical Framework
- **Bayesian Inference**: Probability updates using Bayes' theorem
- **Free Energy Calculation**: Kullback-Leibler divergence minimization
- **Information Theory**: Entropy and mutual information for decision-making
- **Game Theory**: Nash equilibrium and optimal strategies

## 🎯 Usage Examples

### 1. Simulation Mode
```python
# Run automated simulation
python examples/blackjack_demo.py
# Choose option 1 for AI vs Basic Strategy comparison
```

### 2. Interactive Play
```python
# Play against the AI
python examples/blackjack_demo.py
# Choose option 2 for human vs AI gameplay
```

### 3. Custom Agent Development
```python
from agents.blackjack_agent import BlackjackActiveInferenceAgent

# Create custom agent
agent = BlackjackActiveInferenceAgent(
    name="MyAgent",
    bankroll=1000.0,
    risk_tolerance=0.2,
    learning_rate=0.01
)

# Train against other agents
game = BlackjackGame(num_decks=6)
game.add_player(agent)
results = game.play_round()
```

## 📊 Performance Metrics

### Evaluation Criteria
- **Win Rate**: Percentage of winning hands/games
- **Bankroll Growth**: Long-term profitability
- **Risk-Adjusted Returns**: Sharpe ratio equivalent for gambling
- **Adaptation Speed**: Learning curve analysis
- **Strategy Consistency**: Behavioral stability over time

### Benchmarking
- **Basic Strategy**: Mathematical optimal play for comparison
- **Human Players**: Performance against human opponents
- **Monte Carlo**: Statistical validation of strategies
- **Cross-Validation**: Robustness across different conditions

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Quick Start
1. Install dependencies
2. Run the blackjack demo
3. Choose simulation or interactive mode
4. Observe AI behavior and performance

### VS Code Tasks
- **Install Dependencies**: `Ctrl+Shift+P` → "Tasks: Run Task" → "Install Dependencies"
- **Run Demo**: `Ctrl+Shift+P` → "Tasks: Run Task" → "Run Blackjack Demo"
- **Run Tests**: `Ctrl+Shift+P` → "Tasks: Run Task" → "Run Tests"

## 🔮 Future Enhancements

### Planned Features
1. **Multi-Agent Tournaments**: Agents competing in brackets
2. **Evolutionary Strategies**: Genetic algorithm for strategy optimization
3. **Real-Time Adaptation**: Online learning during gameplay
4. **Explainable AI**: Interpretable decision-making processes
5. **Mobile Interface**: Web-based gameplay interface

### Research Directions
- **Hierarchical Active Inference**: Multi-level decision architectures
- **Social Cognition**: Theory of mind for opponent modeling
- **Curiosity-Driven Learning**: Intrinsic motivation for exploration
- **Meta-Learning**: Learning to learn new games quickly

## 🤝 Contributing

### Development Guidelines
1. Follow active inference principles in agent design
2. Implement proper Bayesian inference for belief updates
3. Include comprehensive testing and documentation
4. Maintain consistency with Friston's and Levin's frameworks

### Code Style
- Type hints for all functions and classes
- Comprehensive docstrings with examples
- Unit tests for core functionality
- Performance profiling for optimization

## 📚 References

### Key Papers
1. Friston, K. (2010). The free-energy principle: a unified brain theory?
2. Levin, M. (2019). The Computational Boundary of a "Self"
3. Parr, T. & Friston, K. (2017). Working memory, attention, and salience in active inference
4. Fountas, Z. (2020). Deep active inference agents using Monte-Carlo methods

### Implementation Resources
- [Active Inference Tutorial](https://github.com/pymdp/pymdp)
- [Bayesian Deep Learning](https://github.com/JavierAntoran/Bayesian-Neural-Networks)
- [Game Theory in AI](https://github.com/topics/game-theory)

---

*This project represents a cutting-edge fusion of computational neuroscience, artificial intelligence, and game theory, creating agents that don't just play games but truly understand them through principled probabilistic reasoning.*
