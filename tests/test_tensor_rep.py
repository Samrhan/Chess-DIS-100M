from src.tensor_rep import fen_to_tensor, tensor_to_fen_visual
import chess
import numpy as np

def test_tensor_rep():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tensor = fen_to_tensor(fen)

    print("Tensor shape:", tensor.shape)
    assert tensor.shape == (19, 8, 8)

    # Verify pieces
    visual = tensor_to_fen_visual(tensor)
    print("Reconstructed Visual Board:")
    print(visual)

    expected_visual = \
"""rnbqkbnr
pppppppp
........
........
........
........
PPPPPPPP
RNBQKBNR"""

    assert visual.strip() == expected_visual.strip()

    # Verify turn
    assert np.all(tensor[12] == 1.0) # White to move

    # Verify castling
    assert np.all(tensor[13] == 1.0)
    assert np.all(tensor[14] == 1.0)
    assert np.all(tensor[15] == 1.0)
    assert np.all(tensor[16] == 1.0)

    # Verify ep
    assert np.sum(tensor[17]) == 0

    # Verify 50 move
    assert np.all(tensor[18] == 0.0)

    print("Test passed!")

if __name__ == "__main__":
    test_tensor_rep()
