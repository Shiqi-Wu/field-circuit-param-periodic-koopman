# Hard-Constraint Koopman Operator for Periodic Dynamics

This repository implements and experiments with a parameterized Koopman Operator tailored to systems that exhibit periodic dynamics. The key idea is to hard-constrain the Koopman operator's structure so that it enforces periodic (rotational) spectral properties: the linear operator is built from block-diagonal 2×2 rotation blocks (cos/sin), guaranteeing eigenvalues on the unit circle for the rotational subspaces.

This approach helps keep learned models physically consistent for oscillatory systems (for example, circuits with inductance/capacitance, or other oscillators) by preventing spurious exponential growth or decay that a fully unconstrained linear operator might produce.

## High-level overview

- States are encoded into a learnable dictionary using `TrainableDictionary` (implemented in `src/resnet.py`). The dictionary output concatenates a constant term, the raw state, and a learned encoding.
- A neural network (ResNet) maps system parameters (e.g., material parameters, driving frequency) into parameters of the Koopman operator:
  - For periodic components, the network outputs angles (or direct cos/sin) which are used to form 2×2 rotation blocks. Stacking these blocks on the diagonal yields a block-diagonal Koopman matrix K whose rotational subspaces have spectral radius 1.
  - Optionally the network also outputs a V matrix (a linear mapping in dictionary space) and an input coupling matrix B.
- Prediction is performed in dictionary space using the linear update: x_dic_{t+1} = x_dic_t * K + u_dic_t * B. The implementation uses batched matrix multiplications and supports parameter-conditioned K and B.
- The loss combines mean-squared-error between predicted and true dictionary-space trajectories and a regularization that penalizes pathological V matrices (via an approximate inverse-norm / condition number term).

## Key files and modules

- `src/param_periodic_koopman.py`
  - Implements the main parameterized modules:
    - `ParamBlockDiagonalMatrix` / `ParamBlockDiagonalMatrixWoV`: Build block-diagonal rotation K (and optionally V) from a ResNet output. This is the core "hard-constraint" module.
    - `ParamBlockDiagonalKoopmanWithInputs`: End-to-end module combining dictionary, A (K/V), and B for systems with inputs.
    - `ParamBlockDiagonalKoopmanWithInputs3NetWorks`: Variant that predicts Lambda (rotations), V, and B via three separate networks for more flexible training.
    - Other utility modules: `ParamSkewSymmetricMatrix`, `ParamSpecialOrthogonalMatrix`, `ParamBlockTriangularMatrix` for building structured matrices.
- `src/periodic_koopman.py`
  - Contains a simpler block-diagonal Koopman example (`BlockDiagonalKoopman`, `TimeEmbeddingBlockDiagonalKoopman`) demonstrating a minimal implementation of rotation blocks and dictionary-based time embedding.
- `src/train_param_periodic.py`
  - Training loop, dataset loading, loss computation, and checkpointing. Training uses MSE in dictionary space plus the V regularization.
- `src/args.py`
  - CLI argument parser (`--config`) and YAML configuration loader.
- `src/resnet.py`
  - Network building blocks and `TrainableDictionary` implementation. `TrainableDictionary` supports different encoder types (ResNet or Transformer-like encoder) and returns [1, x, encoded] concatenation.
- `src/data.py`
  - Data loading utilities: reads `.npy` files (each expected to contain `data`, `params`, `inputs`), applies normalization and PCA, slices trajectories for training sequences, and returns PyTorch DataLoaders.

## Design decisions and assumptions

- The hard constraint on K (block-diagonal 2×2 rotations) enforces periodic behavior for the subspaces associated to those blocks. This is appropriate when periodic degrees of freedom can be separated into independent 2D rotational subspaces.
- The dictionary includes raw state + learned features + a constant term, enabling affine reconstructions and input injection.
- V is learned (or predicted) in some variants; to avoid numerical instability, the code applies a regularization term approximating inverse norms / condition number of V.

## How to run

1. Install dependencies (choose appropriate torch matching your CUDA/CPU environment):

```bash
pip install torch numpy pyyaml tqdm scikit-learn matplotlib
```

2. Train with an example config (adjust path to your config file):

```bash
python src/train_param_periodic.py --config configs/experiment_1.yaml
```

Key config keys are in each `configs/*.yaml` file and typically include: `save_dir`, `data_dir`, `dictionary_dim`, `A_layers`, `B_layers`, `u_layers`, `epochs`, `lr`, `sample_step`, `pca_dim`, `batch_size`, `validation_split` etc.

Outputs in `save_dir` include:
- `model_state_dict.pth` — trained model weights
- `dataset.pth` — saved dataset object
- `losses.pth`, `log.txt` — training logs and loss histories

## Regularization and numerical stability

- The code contains `regularization_loss` methods which compute a proxy for V's condition number and apply a log/ReLU-style penalty to avoid extremes. This helps prevent V from becoming numerically singular during training.

## Limitations and extensions

- The assumption of independent 2D rotational subspaces might not hold if the system has strong couplings between periodic modes. In those cases, consider:
  - Allowing off-diagonal coupling (e.g., `ParamBlockTriangularMatrix` or learned upper-triangular blocks).
  - Predicting a full special orthogonal matrix (via matrix exponential of a skew-symmetric generator) and combining it with learned scaling.
- Improve V regularization with explicit condition-number minimization, spectral norm penalties, or by parameterizing V with orthonormal bases plus scaling.