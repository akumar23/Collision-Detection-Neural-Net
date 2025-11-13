# Collision Detection Neural Net

A neural network-based collision detection and path planning system for autonomous robots. The robot uses sensor readings and a trained neural network to predict safe actions and navigate toward goals while avoiding collisions.

## Project Structure

```
Collision-Detection-Neural-Net/
├── src/
│   └── robot_navigation/      # Core package modules
│       ├── __init__.py
│       ├── simulation.py       # Simulation environment and robot physics
│       ├── steering.py         # Steering behaviors (Seek, Wander)
│       ├── networks.py         # Neural network architecture
│       ├── data_loaders.py     # Data loading and preprocessing
│       └── helper.py           # Utility functions
├── scripts/                    # Executable scripts
│   ├── run.py                 # Main simulation runner
│   ├── train.py               # Model training script
│   └── collect_data.py       # Training data collection
├── tests/                     # Test files
│   └── test_robot_movement.py # Automated testing and evaluation
├── data/                      # Training data
│   └── training_data.csv
├── models/                    # Saved models and scalers
│   ├── saved_model.pkl
│   └── scaler.pkl
├── results/                   # Test results and statistics
├── assets/                    # Images and media
│   ├── robot.png
│   ├── robot_inverse.png
│   └── demo.gif
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11.5 (tested version, may work with other versions)
- See `requirements.txt` for package dependencies

## Installation

1. Clone the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Simulation

To run the robot simulation with the trained model:

```bash
python scripts/run.py
```

### Training the Model

To train or retrain the neural network model:

```bash
python scripts/train.py
```

The model will be saved to `models/saved_model.pkl` and the scaler to `models/scaler.pkl`.

### Collecting Training Data

To collect new training data:

```bash
python scripts/collect_data.py
```

This will generate `data/training_data.csv` with sensor readings and collision data.

### Running Tests

To run automated tests and evaluate robot performance:

```bash
# Single test
python tests/test_robot_movement.py

# Multiple tests with custom parameters
python tests/test_robot_movement.py --tests 10 --iterations 2000

# Quiet mode
python tests/test_robot_movement.py --quiet
```

Test results will be saved to the `results/` directory.

## Features

- **Neural Network-Based Collision Prediction**: Uses a trained neural network to predict collision probability for different actions
- **Adaptive Threshold System**: Dynamically adjusts collision threshold based on robot's progress and situation
- **Goal-Seeking Behavior**: Implements seek steering behavior to navigate toward goals
- **Stuck Detection and Recovery**: Automatically detects when the robot is stuck and takes corrective action
- **Comprehensive Testing**: Automated test suite with detailed statistics and performance metrics

## Performance

The current model achieves approximately **75-80% success rate** in reaching goals, with:
- Average iterations per test: ~800-1300
- Adaptive collision avoidance
- Efficient path planning

## Demo

Demo video of the current status of the project: ![here](assets/demo.gif)
