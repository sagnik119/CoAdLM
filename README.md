# CorDA-Style Context-Oriented Decomposition with SVD + Covariance

This code demonstrates **CorDA's key novelty**:
1. **Gather** representative samples (context) for the fine-tuning task or for QA (if preserving knowledge).
2. **Compute** activations' covariance for each linear layer and multiply by the layer's weights.
3. **Perform SVD** on the resulting matrix.
4. **Select** either the smallest-r or largest-r singular components, creating an adapter subspace.
5. **Insert** these subspace parameters back into the model as trainable adapters, freezing the rest of the model.

Run `main.py` to:
- Train a baseline Transformer on Shakespeare (source domain).
- Collect a small set of adaptation samples for either knowledge-preserved or instruction-previewed mode.
- Perform CorDA decomposition + adapter insertion.
- Fine-tune only these adapters.
- Compare performance vs. the baseline.

**NOTE:** This is an approximation/simplification of the official CorDA approach. For production usage, see the [iboing/CorDA](https://github.com/iboing/CorDA) repository.

## Requirements
- Python 3.x
- PyTorch
- tqdm
- matplotlib
- requests

## Repository Structure
- `README.md`: This file.
- `adapter.py`: Defines the context-aware adapter module with configurable hyperparameters.
- `model.py`: Contains model definitions, including both the baseline TransformerLM and the TransformerLMWithAdapter.
- `train.py`: Contains training and evaluation routines.
- `utils.py`: Contains data download, preprocessing, and helper functions.
- `main.py`: Main script that runs baseline training, the two fine-tuning experiments, and the ablation study.

## Usage
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd transformer_adapter_comparative

2. Run the main script:
   python main.py