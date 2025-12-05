# Chess-TRM: Data Pipeline & DIS Prep

This repository implements **Phase 1** of the Chess-TRM project, focusing on generating the `Chess-DIS-100M` dataset. It builds a robust pipeline to convert chess games (PGN) into tensor representations (Input X) and multi-depth Stockfish evaluations (Target Y).

## Project Overview

The goal is to train a Tiny Recursive Model (TRM) for chess. This phase handles:
1.  **Infrastructure**: Setting up Stockfish 16+ as an oracle.
2.  **Action Space**: Mapping ~1968 UCI moves to integers.
3.  **Tensor Representation**: Converting FEN strings to (19, 8, 8) tensors.
4.  **ETL Pipeline**: Processing PGN files into HDF5 datasets.

## Installation

### Prerequisites
*   Python 3.10+
*   Stockfish 16+ (installed and available in PATH or `/usr/games/stockfish`)

### Install Dependencies
This project uses `uv` for package management.

```bash
uv sync
```
Or with standard pip:
```bash
pip install .
```

### Installing Stockfish (Linux)
```bash
sudo apt-get update && sudo apt-get install stockfish
```

## Usage

### 1. Build Dataset
To process a PGN file and generate the HDF5 dataset:

```bash
python3 src/build_dataset.py input_games.pgn output_dataset.h5
```

Optional arguments:
*   `--max_games N`: Limit processing to N games.

### 2. Run Tests
The project uses `pytest` for testing.

```bash
pytest
```

## Module Details

### Module 1: Engine Handler (`src/engine_handler.py`)
Wraps `python-chess` to communicate with Stockfish.
*   **Key Method**: `analyze_position(fen, depths=[1, 2, ...])`
*   **Outputs**: CP Score (clipped [-1500, 1500]), Best Move, Policy Distribution (MultiPV).

### Module 2: Action Space (`src/action_space.py`)
Manages the vocabulary of legal chess moves.
*   **Vocabulary Size**: 1968 unique moves.
*   **Artifact**: `move_to_id.json` (generated on first run).

### Module 3: Tensor Representation (`src/tensor_rep.py`)
Converts board state to numeric tensors.
*   **Shape**: (19, 8, 8) - Channels First.
*   **Channels**:
    *   0-11: Pieces (White/Black P, N, B, R, Q, K)
    *   12: Turn
    *   13-16: Castling Rights
    *   17: En Passant
    *   18: Rule 50 counter (normalized)

### Module 4: Pipeline (`src/build_dataset.py`)
Ingests PGNs, applies filters (skip opening, mates), queries the engine, and writes to HDF5.
*   **Output Format**:
    *   `input_tensor`: uint8 (N, 8, 8, 19) (Transposed for storage)
    *   `target_dX_move`: int16
    *   `target_dX_score`: float16

## Development

A `.devcontainer` configuration is provided for VS Code users, which sets up a Python environment with Stockfish installed.
