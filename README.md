## **Bias-Restrained Prefix Representation Finetuning for Mathematical Reasoning**

### Overview
- BREP focuses on modifying intermediate activations of large language models through prefix tuning, ReFT, and LoRA to improve reasoning, factuality, and robustness.
- The project offers unified training, inference, and analysis scripts so experiments can be reproduced across different base models (e.g., Llama, Qwen) and datasets.

### Directory Layout
- `model.py`: Defines the pluggable `ActivationModel` and `ActivationLayer` that inject trainable parameters at FFN, residual, or other positions.
- `Prefix/`: Training, inference, and evaluation scripts for prefix control and ReFT (e.g., `prefix_train.py`, `answer.py`, `eval.py`).
- `Numprob/`: Numerical intervention and probing utilities (e.g., `analysis.py`, `prober.py`, `reft_intervene.py`).
- `Truthful/`: Evaluation and visualization scripts for factuality and robustness metrics.
- `lora/`: LoRA training and evaluation entry points.
- `bash/`: Bash wrappers for end-to-end workflows (`run.sh`, `analyze.sh`, `faithful.sh`, etc.).
- `requirements.txt`: Python dependency list.
- `template.py` and helpers: Shared inference templates, prompts, and experiment configs.

### Environment Setup
- Use **Python 3.10+** and install dependencies inside an isolated virtual environment.
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- For GPU acceleration, ensure CUDA, `torch`, `xformers`, `deepspeed`, and other libraries are version-compatible. Several packages (`trl`, `peft`) are installed from Git, so outbound GitHub access is required.

### Quick Start
- **Prefix training**: Adjust parameters under `condition=train` in `bash/run.sh` (model name, dataset path, device, etc.), then run:
```
bash bash/run.sh train
```
- **Inference and evaluation**: Configure weight paths and datasets under `condition=brep` or `condition=eval`, then execute:
```
bash bash/run.sh brep
bash bash/run.sh eval
```
- **Other workflows**: `bash/numprob.sh`, `bash/faithful.sh`, and `bash/analyze.sh` launch numerical probes, truthful statistics, and result analysis respectively.

### Data and Outputs
- Default datasets live under `dataset/` (e.g., `dataset/prm800k/`, `dataset/math10k/`); prepare or replace them according to your environment.
- Training and evaluation artifacts are written to `Results/` (e.g., `Results/BREP/...`, `Results/ReFT/...`) and can feed directly into the visualization scripts.

### Workflow Notes
- Scripts in `Prefix/`, `Numprob/`, and other submodules expose command-line arguments for model selection, datasets, and hyperparameters. Use `--help` for detailed options.
- `ActivationLayer` supports multiple layer types (`all`, `scaling`, `bias`, `ln`) and insertion points (such as `ffn_up`, `res`); keep these aligned with the training script arguments.
- Store training logs and checkpoints in dedicated directories so they can be compared and plotted using utilities in `Truthful/` and `Numprob/`.

### Extending the Project
- When adding new models or tasks, reuse the bash templates and create scripts in the corresponding subdirectory to keep the workflow consistent.
- If installation or runtime issues arise, capture the command and logs to speed up debugging.