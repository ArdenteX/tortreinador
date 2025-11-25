from openpyxl.styles.builtins import output

# Torch Treinador

**Torch Treinador** is a flexible and highly customizable training framework designed for PyTorch. It aims to eliminate the repetitive boilerplate code associated with training loops while giving developers complete control over the computation logic.

By decoupling the **Training Loop** (iteration, logging, checkpointing) from the **Computation Logic** (forward pass, loss calculation), Torch Treinador allows you to focus purely on model architecture and mathematical operations.

## Key Features

### 1. Customizable Calculation Layer
Unlike rigid wrapper libraries, Torch Treinador does not hide the model's forward pass. You can easily customize the training behavior by overriding the `calculate` method. Whether you are dealing with multi-input models, complex loss functions (e.g., GANs, MDNs), or custom gradient operations, simply define **how** to calculate the loss, and the trainer handles **when** to run it.

### 2. Powerful Metric Manager
Tracking model performance goes beyond just monitoring loss. With the built-in `MetricManager`, you can define, track, and log an unlimited number of evaluation metrics (such as R2 Score, Accuracy, MAE) simultaneously. The trainer automatically integrates these metrics into the progress bar, TensorBoard logs, and CSV reports without cluttering your code.

### 3. Comprehensive Training Utilities
Torch Treinador comes "batteries included" with essential tools for modern deep learning:
* **Smart Checkpointing:** Automatically save the best models based on Loss, specific Metrics, or a combination of both.
* **Advanced Scheduling:** Built-in support for Warmup, Cosine Annealing, and Multi-step Learning Rate decay.
* **Flexible Logging:** Support for in-memory recording, CSV file logging, and TensorBoard visualization.
* **Overfitting Prevention:** Integrated mechanisms to stabilize training.


## Installation
This package needs Python>=3.7 and the version of Pytorch used in development is 2.5.1 and cuda12.4, considering the different version of cuda, the package will
not install Pytorch automatically. You should check your cuda's version, install the suitable [pytorch](https://pytorch.org/get-started/previous-versions/) first. Then, run the command below:
```
pip install tortreinador 
```
## Quick Start

### 1. Data Preparation
Tortreinador simplifies data loading and preprocessing. You can load data directly from a file (e.g., Excel) or convert existing tensors into Dataloaders.

**Option A: Load from file (Automatic Preprocessing)**
```python
from tortreinador.utils.preprocessing import load_data, ScalerConfig
import pandas as pd

# Load your dataset
data = pd.read_excel('your_data.xlsx')

# Define parameters
input_cols = ['Feature1', 'Feature2', 'Feature3']
output_cols = ['Target1', 'Target2']

# ScalerConfig controls normalization (e.g., 'standard' or 'minmax')
scaler_config = ScalerConfig(on=True, method='standard', normal_y=True)

# Get Dataloaders automatically
t_loader, v_loader, test_x, test_y, s_x, s_y = load_data(
    data=data, 
    input_parameters=input_cols,
    output_parameters=output_cols,
    normal=scaler_config, 
    if_shuffle=True, 
    batch_size=1024, 
    num_workers=4
)
```

**Option B: Use existing tensors**
```python
from tortreinador.utils.preprocessing import get_dataloader
import torch

# Assuming you already have tensors
train_loader = get_dataloader(x=train_x_tensor, y=train_y_tensor, batch_size=1024, shuffle=True)
val_loader = get_dataloader(x=val_x_tensor, y=val_y_tensor, batch_size=1024, shuffle=False)
```

### 2. Define Model, Metrics and customize the calculation layer
Initialize your PyTorch model, define the metrics you want to track using `MetricManager`, and rewrite the `calculate`

Important: Ensure the order of values in `update_values` matches the order of metrics defined in your MetricManager.
```python
import torch.nn as nn
import torch.optim as optim
from tortreinador.utils.Recorder import MetricManager, MetricDefine
from tortreinador.train import TorchTrainer
from tortreinador.utils.metrics import r2_score

# 1. Setup standard PyTorch components
model = YourCustomModel()  # Your nn.Module
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. Define Metrics (The core of Torch Treinador)
# You can register unlimited metrics. 'metric_mode' usually maps to internal logic (e.g., 0 for simple averaging).
mm = MetricManager([
    MetricDefine(metric_name='Loss', metric_mode=0, use_as_criterion=True),
    MetricDefine(metric_name='R2_Score', metric_mode=0, use_as_baseline=True)
])

class CustomTrainer(TorchTrainer):
    def calculate(self, x, y, mode=1):
        """
        x: input batch
        y: target batch
        mode: 1 for training, 2 for validation (handled by the framework)
        """
        # Custom Forward Pass (e.g., for MLP)
        y_pred = self.model(x)

        # Custom Loss Calculation
        loss = self.criterion(y, y_pred)

        # Custom Metric Calculation
        r2_val = r2_score(y, y_pred)

        # Return values must match the MetricManager order: [Loss, R2]
        metric_return = [loss, r2_val]

        return self._standard_return(mode=mode, update_values=metric_return)

# 3. Initialize Trainer
trainer = CustomTrainer(
    is_gpu=True, 
    epoch=50, 
    optimizer=optimizer, 
    model=model, 
    criterion=criterion, 
    metric_manager=mm  # Pass the manager here
)
```

### 3. Configuration and Training
Use `config_generator` to control training hyperparameters like warmup, checkpoint saving, and validation cycles.
```python
from tortreinador.train import config_generator

# Generate configuration
# This sets up saving paths, warmup epochs, and auto-save conditions
config = config_generator(
    model_save_path="./resources/checkpoints/", 
    warmup_epochs=5, 
    best_metric=0.8,  # Threshold for saving best model
    auto_save=10, 
    validation_cycle=1
)

# Start Fitting!
# Pass the dataloaders and unpack the configuration
result = trainer.fit(t_loader, v_loader, **config)
```

## Future Works

### Grid Search function and Event System are developing

## Functions
Please visit https://ardentex.github.io/tortreinador/












