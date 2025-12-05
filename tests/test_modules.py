import numpy as np
import h5py
import os
import pytest
from src.tensor_rep import fen_to_tensor
from src.action_space import ActionSpace
from src.engine_handler import EngineHandler
from src.build_dataset import flush_to_h5

def test_reversibility():
    # We can't fully reverse fen_to_tensor because we lose move counters (fullmove)
    # but we can verify piece placement.
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tensor = fen_to_tensor(fen)

    # Check specific pieces
    # White Rooks at (0,0) and (0,7) -> channel 3 (piece map: P=0,N=1,B=2,R=3,Q=4,K=5)
    # tensor shape (19, 8, 8)

    # Check A1 (rank 0, file 0)
    assert tensor[3, 0, 0] == 1.0 # White Rook
    # Check A8 (rank 7, file 0) -> Black Rook -> Channel 6+3=9
    assert tensor[9, 7, 0] == 1.0

def test_oracle_consistency():
    handler = EngineHandler()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    try:
        res = handler.analyze_position(fen, depths=[1, 2])

        # Check structure
        assert 1 in res
        assert 2 in res
        assert len(res[1]["best_move"]) >= 4 # e.g. e2e4
    finally:
        handler.close()

def test_action_space():
    space = ActionSpace()
    move = "e2e4"
    idx = space.encode_move(move)
    decoded = space.decode_move(idx)
    assert move == decoded

def test_h5_output(tmp_path):
    # Create a temporary H5 file
    output_file = tmp_path / "test_dataset.h5"

    # Create dummy buffer
    buffer = {
        "input_tensor": [np.zeros((19, 8, 8), dtype=np.float32)],
        "target_d1_move": [0],
        "target_d1_score": [0.5],
        "target_d5_move": [0],
        "target_d5_score": [0.5]
    }

    # Flush to file
    flush_to_h5(str(output_file), buffer)

    # Verify content
    assert output_file.exists()
    with h5py.File(output_file, "r") as f:
        assert "input_tensor" in f
        assert f["input_tensor"].shape == (1, 8, 8, 19)
        assert "target_d1_move" in f
