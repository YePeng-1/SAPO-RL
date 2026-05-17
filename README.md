# SAPO-RL: Sparse Adaptive Optimization for Reinforcement Learning

## Overview

This repository contains test scripts and trained agents for evaluating and comparing different reinforcement learning and optimization algorithms on fuselage actuator adjustment tasks using ANSYS simulations. 

## Algorithms Included

- **D3QN**: Double Dueling Deep Q-Network
- **PPO**: Proximal Policy Optimization
- **ReEs**: Neural Network-based Reinforcement Learning
- **SL**: Sparse Learning with ADMM

## Directory Structure

```
SAPO-RL/
├── Agents/                  # Trained model weights (.pth, .pt files)
├── AssemblyGym/             # Reinforcement learning environments
│   └── envs/
│       └── FuselageActuators/
│           ├── AnsysFiles/  # ANSYS input files (Training, Testing, Benchmarking)
│           ├── Shapes/       # Pre-computed initial shapes
│           ├── Surrogates/   # Surrogate models for simulation
│           └── FuselageActuators_env_v22.py
├── Data/                    # Training and test results
│   ├── D3QN_error.csv
│   ├── D3QN_Max.csv
│   ├── D3QN_test_error.csv
│   ├── D3QN_test_Max.csv
│   ├── PPO_error.csv
│   ├── PPO_Max.csv
│   ├── PPO_test_error.csv
│   ├── PPO_test_Max.csv
│   ├── ReEs_error.csv
│   ├── ReEs_Max.csv
│   ├── ReEs_loss.csv
│   ├── ReEs_test_error.csv
│   ├── ReEs_test_loss.csv
│   ├── ReEs_test_Max.csv
│   ├── ADMM_Benchmark_Forces.csv
│   ├── 0208/, 0325/, 0409/, 1102/  # Historical test results by date
│   └── 025.csv, 030.csv, 035.csv, 040.csv, 045.csv, 050.csv
├── Ansys_test_D3QN.py         # Test script for D3QN agent
├── Ansys_test_PPO.py          # Test script for PPO agent
├── Ansys_test_ReEs.py          # Test script for ReEs agent
├── Ansys_test_SL.py            # Test script for Sparse Learning agent
└── ADMM_Sbeta.py             # ADMM optimization solver for sparse learning
```

## Requirements

- Python 3.8+
- PyTorch
- numpy
- pandas
- matplotlib
- scikit-learn
- joblib
- ansys-mapdl (for ANSYS simulation)
- gym

## Usage

### Running ANSYS Tests

Each test script evaluates a different algorithm on ANSYS simulations:

```bash
# Test D3QN agent
python Ansys_test_D3QN.py

# Test PPO agent
python Ansys_test_PPO.py

# Test ReEs agent
python Ansys_test_ReEs.py

# Test Sparse Learning agent
python Ansys_test_SL.py
```

### Testing with Different Parameters

You can test algorithm performance with different max deviation limitations:

```bash
python Ansys_test_D3QN.py --n_actions 10
python Ansys_test_PPO.py --n_actions 10
python Ansys_test_ReEs.py --n_actions 10
python Ansys_test_SL.py --n_actions 10
```

### Visualization and Statistical Analysis

Generate training/test performance curves and statistical comparisons:

```bash
# Draw training and test curves
python Draw_Figure.py

# Compare algorithm performance with statistical tests
python Compare_Results.py
```

## Test Scripts Features

- **ANSYS Integration**: Direct connection to ANSYS for accurate simulation
- **Real-time Visualization**: Display initial and adjusted shapes with force vectors
- **Performance Metrics**: Initial Error, Final Error, Max Deviation, Actuator Number
- **Statistical Analysis**: Paired hypothesis tests (t-test or Wilcoxon) for algorithm comparison
- **Data Logging**: Results saved to tensorboard and CSV files for post-processing

## Model Checkpoints

The `Agents/` folder contains trained model checkpoints:

- `SequentialFuselageActuators__D3QN_SequentialFuselageActuators__1__1744185882.pth` - D3QN checkpoint
- `agent_32767872steps.pt` - PPO checkpoint
- `SequentialFuselageActuators__NN_SequentialFuselageActuators__1__1744199521.pth` - ReEs checkpoint

Sparse Learning does not require pretrained checkpoints as it uses online optimization.

## Data Files

The `Data/` folder contains historical training and test results:

- **Training Data**: Error curves and max deviation progress for D3QN, PPO, ReEs
- **Test Data**: Performance metrics for various test cases
- **Parameter Sweeps**: Results for different max deviation limitation parameters (0.025, 0.030, 0.035, etc.)
- **Benchmark Data**: ADMM baseline benchmark forces

## Notes

- All test scripts require ANSYS license and proper connection configuration
- The `AssemblyGym` environment is adapted from the main FuselageActuator project
- Model files are not automatically downloaded; they must be present in the `Agents/` folder
- CSV data files are optional for visualization scripts only

## License

This project is for research purposes. Please cite appropriately if used in publications.