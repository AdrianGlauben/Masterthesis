# Replacing PUCT with a Planning Model
This repository was build for my Mastersthesis at TU Darmstadt. It contains an implementation for
replacing PUCT in AlphaZero with a learnable Planning Model.
Our AlphaZero implementation for Connect4 is based on the implementation by Soh (2016), which can be found here: https://towardsdatascience.com/from-scratch-implementation-of-alphazero-for-connect4-f73d4554002a.

## Usage
### Step 1
Use the main_pipeline.py to train a AlphaZero base model for the game of Connect4 by executing it with
the chosen hyperparameters.

### Step 2
Initialize the planning model by first generating data with pm_data_generation.py, then use this data to initialize
the various planning models using pm_train.py

### Step 3
After placing the initialized planning models as well as the AlphaZero base model in the following folder structure,
execute the main_pipeline_ppo.py to optimize the planning model.

Base model directory: './src/data/ppo_data/a0_base_model.pth.tar'

Planning model directories: f'./src/data/ppo_data/pm_model_data/{pm_id}/{pm_id}_iter_0.pth.tar'

Possible pm_ids: ['SPM_base', 'SPM_QVar', 'ConvPM_base', 'ConvPM_QVar', 'ConvPM_MH', 'ConvPM_All']
