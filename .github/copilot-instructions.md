# Copilot Instructions for Active Inference Games

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Overview
This is an active inference gaming system that implements Friston's active inference framework combined with Michael Levin's ideas about intelligence and agency. The project focuses on creating intelligent agents that can play games like blackjack and Texas Hold'em poker.

## Key Concepts to Remember
- **Active Inference**: Implement Friston's free energy principle and predictive processing
- **Agency**: Incorporate Michael Levin's concepts of goal-directed behavior and intelligence
- **Game Theory**: Apply optimal strategy concepts for card games
- **Multi-Agent Systems**: Agents can play against each other and humans
- **PyTorch**: Use for neural network implementations and learning
- **Bayesian Inference**: Core to active inference framework

## Code Style Guidelines
- Follow Python best practices with type hints
- Use dataclasses for game states and actions
- Implement clean abstractions for games and agents
- Include comprehensive docstrings explaining active inference concepts
- Use descriptive variable names that reflect active inference terminology (beliefs, priors, posteriors, etc.)

## Active Inference Terminology
- **Beliefs**: Agent's probabilistic representations of the world
- **Priors**: Initial beliefs before observing evidence
- **Posteriors**: Updated beliefs after observing evidence
- **Free Energy**: Objective function to minimize (surprise + complexity)
- **Generative Model**: Agent's model of how observations are generated
- **Inference**: Process of updating beliefs based on observations
- **Action Selection**: Choosing actions to minimize expected free energy

## Game-Specific Considerations
- **Blackjack**: Focus on card counting, risk assessment, and optimal stopping
- **Texas Hold'em**: Emphasize opponent modeling, bluffing, and strategic betting
- **Human Interaction**: Design intuitive interfaces for human players
- **Multi-Agent**: Implement communication and coordination between agents

## Dependencies to Use
- torch (PyTorch for neural networks)
- numpy (numerical computations)
- scipy (statistical functions)
- matplotlib (visualization)
- typing (type hints)
- dataclasses (structured data)
- enum (game states and actions)
- random (game randomization)
- abc (abstract base classes)
