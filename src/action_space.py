import json
import chess
import os

class ActionSpace:
    def __init__(self, vocab_path="move_to_id.json"):
        self.vocab_path = vocab_path
        self.move_to_id = {}
        self.id_to_move = {}

        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                self.move_to_id = json.load(f)
            self.id_to_move = {v: k for k, v in self.move_to_id.items()}
        else:
            self._generate_vocabulary()
            self._save_vocabulary()

    def _generate_vocabulary(self):
        moves = set()

        # Iterate over all squares for regular moves (Queen + Knight coverage)
        for src in range(64):
            src_rank, src_file = divmod(src, 8)

            for dst in range(64):
                if src == dst: continue
                dst_rank, dst_file = divmod(dst, 8)

                # Check if valid Queen move (same rank, file, or diagonal)
                is_queen_move = (src_rank == dst_rank) or (src_file == dst_file) or (abs(src_rank - dst_rank) == abs(src_file - dst_file))

                # Check if valid Knight move
                d_rank = abs(src_rank - dst_rank)
                d_file = abs(src_file - dst_file)
                is_knight_move = (d_rank == 2 and d_file == 1) or (d_rank == 1 and d_file == 2)

                if is_queen_move or is_knight_move:
                    moves.add(chess.Move(src, dst).uci())

        # Promotions
        # White (rank 6->7) and Black (rank 1->0)
        # Directions: push, capture left, capture right

        # Helper for promotions
        def add_promotions(src_rank, dst_rank):
            for file in range(8):
                src = 8 * src_rank + file

                targets = []
                # Push
                targets.append(8 * dst_rank + file)
                # Capture Left
                if file > 0: targets.append(8 * dst_rank + (file - 1))
                # Capture Right
                if file < 7: targets.append(8 * dst_rank + (file + 1))

                for dst in targets:
                    for p in ['q', 'r', 'b', 'n']:
                        moves.add(chess.Move(src, dst, promotion=chess.Piece.from_symbol(p).piece_type).uci())

        add_promotions(6, 7) # White
        add_promotions(1, 0) # Black

        sorted_moves = sorted(list(moves))
        self.move_to_id = {move: i for i, move in enumerate(sorted_moves)}
        self.id_to_move = {i: move for i, move in enumerate(sorted_moves)}

    def _save_vocabulary(self):
        with open(self.vocab_path, "w") as f:
            json.dump(self.move_to_id, f, indent=2)

    def encode_move(self, uci_move):
        return self.move_to_id.get(uci_move)

    def decode_move(self, move_id):
        return self.id_to_move.get(move_id)

    def get_vocab_size(self):
        return len(self.move_to_id)
