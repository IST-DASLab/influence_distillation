# Influence Distillation

This repository includes the official implementation of the paper [Efficient Data Selection at Scale via Influence Distillation](https://arxiv.org/abs/2505.19051). The code provides tools for selecting influential training samples that best impact model performance on target data.

## Current Features
- Implementation of JVP (Jacobian Vector Product) embeddings for efficient influence computation
- First-order, unweighted Influence Distillation algorithm + selecting the most influencial samples

## Roadmap
The following updates are planned:
- Code for reproducing paper experiments.
- Batching JVP embeddings computation.
- Chunking the JVP and gradient stores to support larger datasets.
- Improve documentation.

## Installation

### Setup
1. Create and activate a new environment:
```bash
mamba create --name infdist python=3.12.9  # or use conda/venv
mamba activate infdist
```

2. Install dependencies:
```bash
pip install torch==2.5.1
pip install -r requirements.txt
pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git

# Install InfDist package
pip install -e .
```

## Usage Guide

### Prerequisites
Before using the selection tools, ensure you have:
- A trained HuggingFace model (`model`)
- A training dataset loader (`train_loader`)
- A target dataset loader (`target_loader`)
- Optionally, an `optimizer`, if your model is warmed-up (recommended, see Step 0 below).

Note: Both datasets must be tokenized with fields: `input_ids`, `label`, and `attention_mask`

### Step 0: Warm-up (Optional)
Before proceeding with influence computation, we recommend warming up the model by training it on a small random subset of the training data. This warm-up phase helps stabilize gradient computations and leads to more reliable influence weights, as demonstrated in our paper. Using a warmed-up model in the subsequent steps typically produces better results. Refer to Appendix A in [the paper](https://arxiv.org/abs/2505.19051).

### Step 1: Create JVP Embeddings
Compute JVP embeddings for your target dataset:

```python
import infdist

jvp_embeddings = infdist.create_jvp_store(
    model,
    train_loader.dataset,
    num_blocks=4,        # Number of transformer blocks
    num_tangents=2,      # Number of tangent vectors
    proj_dim=4096,       # Projection dimension
    seed=43
)
```

This process generates a 4096-dimensional embedding for each training sample. We recommend saving these embeddings to disk for future reuse, avoiding the computational cost of regenerating them.

While the current implementation processes samples one at a time and iterates over tangent vectors sequentially, future versions will support batched computation over both samples and tangent vectors for improved efficiency. Stay tuned for updates!

### Step 2: Select Influential Samples
Use the following code to select the most influential training samples:

```python
import infdist

selected_idx = infdist.pick_first_order(
    model,
    train_loader,
    target_loader,
    optimizer,                       # Could be None
    embeddings=jvp_embeddings,   # From step 1
    k=10000,                         # Number of samples to select
    proj_subset='down_proj',         # Subset of parameters to use for gradient
    param_mask_numel=1_000_000_000,  # Maximum element count to consider
    num_landmarks=4096,              # Number of landmarks for gradient approximation
    seed=43                          # Random seed for reproducibility
)
```

This process:
1. Approximates training sample gradients using the landmark-based approach
2. Computes influence scores by matching them against target sample gradients
3. Returns indices of the `k` most influential training samples

Note that this command only considers one billion elements of the `down_proj` gradients. For smaller models, you can use all gradients by setting `proj_subset` and `param_mask_numel` to `None`. See Appendix I in [the paper](https://arxiv.org/abs/2505.19051) for more details on gradient masking and projection.

### Step 3: Training
Train your model on the selected subset!


## Reproducing Paper Experiments
We use a fork of [this repo](https://github.com/hamishivi/automated-instruction-selection.git). Stay tuned for details and scripts!
