import numpy as np
import chess

def fen_to_tensor(fen_string):
    """
    Converts a FEN string into an 8x8x19 numpy array (Channels Last) or 19x8x8 (Channels First).
    Spec says: (19, 8, 8) or (8, 8, 19).
    I will use (19, 8, 8) (Channels First) as it's more common in PyTorch.

    19 Channels:
    0-5: White pieces (P, N, B, R, Q, K)
    6-11: Black pieces (p, n, b, r, q, k)
    12: Turn (1 if White, 0 if Black)
    13-16: Castling rights (K, Q, k, q)
    17: En passant target
    18: 50-move rule (count / 100)
    """
    board = chess.Board(fen_string)
    tensor = np.zeros((19, 8, 8), dtype=np.float32)

    # 1. Piece placement (Channels 0-11)
    # Map piece types to channel offsets
    # White: P=1, N=2, B=3, R=4, Q=5, K=6 -> channels 0-5
    # Black: p=1, n=2... -> channels 6-11

    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)

            # According to spec: "Ensure A1 corresponds to index (0,0)..."
            # In chess.SQUARES, 0 is A1.
            # In numpy array[rank][file], rank 0 is bottom (A1..H1) if we interpret it visually?
            # Usually tensor[0, 0] is top-left in images.
            # But for chess, we usually map rank/file directly.
            # Let's verify spec: "A1 corresponds to index (0,0)"?
            # Or "A1 corresponds to index 0"?
            # Let's map rank 0 -> index 0, file 0 -> index 0.

            channel_offset = 0 if piece.color == chess.WHITE else 6
            channel = channel_offset + piece_map[piece.piece_type]

            tensor[channel, rank, file] = 1.0

    # 2. Turn (Channel 12)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # 3. Castling rights (Channels 13-16)
    # K white, Q white, k black, q black
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    # 4. En passant (Channel 17)
    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        tensor[17, rank, file] = 1.0

    # 5. 50-move rule (Channel 18)
    tensor[18, :, :] = board.halfmove_clock / 100.0

    return tensor

def tensor_to_fen_visual(tensor):
    """
    Helper to reconstruct a board string for verification (not full FEN, just pieces).
    """
    rows = []
    for r in range(7, -1, -1):
        row_str = ""
        for f in range(8):
            found = False
            # Check white pieces
            for i, p in enumerate(['P', 'N', 'B', 'R', 'Q', 'K']):
                if tensor[i, r, f] == 1:
                    row_str += p
                    found = True
                    break
            if not found:
                # Check black pieces
                for i, p in enumerate(['p', 'n', 'b', 'r', 'q', 'k']):
                    if tensor[i+6, r, f] == 1:
                        row_str += p
                        found = True
                        break
            if not found:
                row_str += "."
        rows.append(row_str)
    return "\n".join(rows)
