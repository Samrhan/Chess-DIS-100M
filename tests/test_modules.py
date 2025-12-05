import unittest
import numpy as np
import chess
import h5py
import os
from src.tensor_rep import fen_to_tensor
from src.action_space import ActionSpace
from src.engine_handler import EngineHandler

class TestPipeline(unittest.TestCase):
    def test_reversibility(self):
        # We can't fully reverse fen_to_tensor because we lose move counters (fullmove)
        # but we can verify piece placement.
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        tensor = fen_to_tensor(fen)

        # Check specific pieces
        # White Rooks at (0,0) and (0,7) -> channel 4 (piece map: P=0,N=1,B=2,R=3,Q=4,K=5)
        # Wait, in my code: P=0, N=1, B=2, R=3, Q=4, K=5.
        # Plus offset 0 for white. So channel 3.
        # tensor shape (19, 8, 8)

        # Check A1 (rank 0, file 0)
        assert tensor[3, 0, 0] == 1.0 # White Rook
        # Check A8 (rank 7, file 0) -> Black Rook -> Channel 6+3=9
        assert tensor[9, 7, 0] == 1.0

    def test_oracle_consistency(self):
        handler = EngineHandler()
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        res = handler.analyze_position(fen, depths=[1, 2])
        handler.close()

        # Check structure
        self.assertIn(1, res)
        self.assertIn(2, res)
        self.assertTrue(len(res[1]["best_move"]) >= 4) # e.g. e2e4

    def test_action_space(self):
        space = ActionSpace()
        move = "e2e4"
        idx = space.encode_move(move)
        decoded = space.decode_move(idx)
        self.assertEqual(move, decoded)

    def test_h5_output(self):
        if os.path.exists("dataset_test.h5"):
            with h5py.File("dataset_test.h5", "r") as f:
                self.assertTrue("input_tensor" in f)
                self.assertEqual(f["input_tensor"].shape[1:], (8, 8, 19))
                self.assertTrue("target_d1_move" in f)

if __name__ == "__main__":
    unittest.main()
