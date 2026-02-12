# Language and Experience: A Computational Model of Social Learning in Complex Tasks

This repository contains the code for the paper:

**"Language and Experience: A Computational Model of Social Learning in Complex Tasks"**
*Cédric Colas, Tracey Mills, Ben Prytawski, Michael Henry Tessler, Noah Goodman, Jacob Andreas, Joshua Tenenbaum*
CogSci 2025 and ICLR 2026

[Paper link](https://arxiv.org/abs/2509.00074)
[Demo website](https://cedriccolas.com/demos/language_and_experience/)

## Abstract

The ability to combine linguistic guidance from others with direct experience is central to human development, enabling safe and rapid learning in new environments. How do people integrate these two sources of knowledge, and how might AI systems? We present a computational framework that models human social learning as joint probabilistic inference over structured, executable world models given sensorimotor and linguistic data. We make this possible by turning a pretrained language model into a probabilistic model of how humans share advice conditioned on their beliefs, allowing our agents both to generate advice for others and to interpret linguistic input as evidence during Bayesian inference. Using behavioral experiments and simulations across 10 video games, we show how linguistic guidance can shape exploration and accelerate learning by reducing risky interactions and speeding up key discoveries in both humans and models. We further explore how knowledge can accumulate across generations through iterated learning experiments and demonstrate successful knowledge transfer between humans and models---revealing how structured, language-compatible representations might facilitate human-machine collaborative learning. 

## Repository Structure

```
infer-vgdl/
├── src/
│   ├── main.py              # Main experiment loop
│   ├── utils.py             # Utility functions
│   ├── agent/               # Agent implementation
│   │   ├── agent.py         # Main agent class
│   │   ├── communicating/   # Language processing (LLM integration)
│   │   └── thinking/        # Inference and planning
│   ├── game/                # Game environment wrapper
│   ├── vgdl/                # VGDL game engine
│   ├── baselines/           # Baseline agents (DQN, pure LLM)
│   └── analysis/            # Analysis scripts
├── games/                   # Game definitions (VGDL format)
├── data_input/              # Input data (descriptions, prompts)
├── configs/                 # Experiment configurations
├── docs/                    # Documentation
└── run_experiment.py        # Simplified entry point
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (for LLM inference)
- Conda (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ccolas/language_and_experience.git
cd language_and_experience
```

2. Create a conda environment:
```bash
conda create -n language_and_experience python=3.10
conda activate language_and_experience
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install MPI (required for parallel inference):
```bash
conda install -c conda-forge mpi4py openmpi
```

5. (Optional) For LLM-based conditions, set up the model path:
```bash
export MODELS_PATH=/path/to/your/models
```

See [docs/DATA_SETUP.md](docs/DATA_SETUP.md) for detailed instructions on obtaining LLM models.

## Quick Start

### Individual Learning (Experience Only)

Run the model learning from experience alone:

```bash
python run_experiment.py --game beesAndBirds --condition individual --n-lives 15
```

### Social Learning with Language

Run the model with access to a language description:

```bash
python run_experiment.py --game beesAndBirds --condition social --msg-source human --n-lives 15
```

### Available Games

- `avoidGeorge` - Avoid enemies
- `beesAndBirds` - Collect items, avoid hazards
- `preconditions` - Resource management
- `portals` - Teleportation mechanics
- `pushBoulders` - Sokoban-style puzzle
- `relational` - Object relationships
- `plaqueAttack` - Shooting game
- `aliens` - Space invaders variant
- `missile_command` - Defend cities
- `jaws` - Survive shark attacks

## Main Experimental Conditions

| Condition | Description | Command Flag |
|-----------|-------------|--------------|
| Individual | Learning from experience only | `--condition individual` |
| Social (Human) | Learning with human-written descriptions | `--condition social --msg-source human` |
| Social (Model) | Learning with model-generated descriptions | `--condition social --msg-source model` |
| Generational | Chain learning across generations | `--condition chain` |

## Demo

For an interactive demo of the model, visit: https://cedriccolas.com/demos/language_and_experience/ 


## Citation

If you use this code in your research, please cite:

```bibtex
@article{colas2025language,
  title={Language and Experience: A Computational Model of Social Learning in Complex Tasks},
  author={Colas, C{\'e}dric and Mills, Tracey and Prystawski, Ben and Tessler, Michael Henry and Goodman, Noah and Andreas, Jacob and Tenenbaum, Joshua},
  journal={arXiv preprint arXiv:2509.00074},
  year={2025}
}
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by the EU's Horizon 2020 programme via Marie Skłodowska-Curie grant; the NSF; the Department of the Air Force Artificial Intelligence Accelerator/


