import argparse
import chess
import chess.pgn
import numpy as np
import h5py
import os
import time
from src.engine_handler import EngineHandler
from src.tensor_rep import fen_to_tensor
from src.action_space import ActionSpace


def build_dataset(pgn_path, output_path, max_games=None, batch_size=1000):
    # Initialize components
    engine = EngineHandler()
    action_space = ActionSpace()

    # Dataset buffers
    data_buffer = {
        "input_tensor": [],
        "target_d1_move": [],
        "target_d1_score": [],
        "target_d5_move": [],  # Example curriculum depth
        "target_d5_score": [],
    }

    game_count = 0
    pos_count = 0
    start_time = time.time()

    # Open PGN
    with open(pgn_path) as pgn:
        while True:
            if max_games and game_count >= max_games:
                break

            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            game_count += 1
            board = game.board()

            # Iterate through moves
            move_count = 0
            for move in game.mainline_moves():
                # Filter: Exclude opening (moves 1-5 = 10 half-moves)
                if move_count < 10:
                    board.push(move)
                    move_count += 1
                    continue

                # Filter: Check for immediate mate?
                if board.is_checkmate():
                    break

                # Process position
                # 1. Convert to Tensor
                fen = board.fen()
                tensor = fen_to_tensor(fen)  # (19, 8, 8)

                # 2. Analyze
                # Depths: 1 and 5 (as example, spec says multi-depth)
                analysis = engine.analyze_position(fen, depths=[1, 5])

                # 3. Targets
                d1_res = analysis[1]
                d5_res = analysis[5]

                d1_move_id = (
                    action_space.encode_move(d1_res["best_move"])
                    if d1_res["best_move"]
                    else 0
                )  # 0 as fallback/padding?
                d5_move_id = (
                    action_space.encode_move(d5_res["best_move"])
                    if d5_res["best_move"]
                    else 0
                )

                # If move not in vocab, we might have an issue. Or just skip.
                if d1_move_id is None or d5_move_id is None:
                    # e.g. null move or special case?
                    # or engine returned None
                    board.push(move)
                    move_count += 1
                    continue

                data_buffer["input_tensor"].append(tensor)
                data_buffer["target_d1_move"].append(d1_move_id)
                data_buffer["target_d1_score"].append(d1_res["score"])
                data_buffer["target_d5_move"].append(d5_move_id)
                data_buffer["target_d5_score"].append(d5_res["score"])

                pos_count += 1

                # Flush if buffer full
                if len(data_buffer["input_tensor"]) >= batch_size:
                    flush_to_h5(output_path, data_buffer)
                    clear_buffer(data_buffer)
                    print(f"Processed {pos_count} positions...")

                board.push(move)
                move_count += 1

    # Flush remaining
    if len(data_buffer["input_tensor"]) > 0:
        flush_to_h5(output_path, data_buffer)

    engine.close()
    print(f"Finished. Processed {game_count} games, {pos_count} positions.")
    print(f"Total time: {time.time() - start_time:.2f}s")


def flush_to_h5(output_path, buffer):
    # If file doesn't exist, create it. If it does, append.
    # H5py resizeable datasets

    mode = "a" if os.path.exists(output_path) else "w"

    with h5py.File(output_path, mode) as f:
        n_samples = len(buffer["input_tensor"])

        if mode == "w":
            # Create datasets
            # Compression: 'gzip'

            # Input: (N, 19, 8, 8) uint8? Spec says uint8 (compressed).
            # But my tensor is float. I should cast to uint8 if it's 0/1.
            # But rule 50 is float 0..1. So maybe keep float or scale to uint8?
            # Spec: "input_tensor: uint8 (8x8x19)".
            # Channel 18 is rule50 count / 100. If we map to uint8, we lose precision?
            # Or we store rule50 as integer count?
            # The spec says "Normalized: count / 100".
            # If I store as uint8, I can store count directly (0-100) and normalize later.
            # Let's convert tensor to uint8.
            # Channels 0-17 are 0/1. Channel 18 is 0.xx.
            # Let's multiply channel 18 by 100 and cast to uint8.

            # Prepare data
            tensor_data = np.array(buffer["input_tensor"], dtype=np.float32)
            # Normalize Rule50 back to int for storage? Or just store as float16?
            # Spec says "input_tensor: uint8".
            # So I will scale channel 18 by 100.
            tensor_data[:, 18, :, :] *= 100
            tensor_data = np.round(tensor_data).astype(np.uint8)

            # Channels Last or First?
            # Spec says "Dimension: (19, 8, 8) or (8, 8, 19)".
            # Spec for storage: "input_tensor: uint8 (8x8x19)".
            # This implies Channels Last.
            # I am generating (19, 8, 8). I should transpose for storage if spec requires 8x8x19.
            # "input_tensor: uint8 (8x8x19)".

            tensor_data = np.transpose(tensor_data, (0, 2, 3, 1))  # (N, 8, 8, 19)

            f.create_dataset(
                "input_tensor",
                data=tensor_data,
                maxshape=(None, 8, 8, 19),
                compression="gzip",
                chunks=True,
            )

            f.create_dataset(
                "target_d1_move",
                data=buffer["target_d1_move"],
                maxshape=(None,),
                dtype="int16",
            )
            f.create_dataset(
                "target_d1_score",
                data=buffer["target_d1_score"],
                maxshape=(None,),
                dtype="float16",
            )
            f.create_dataset(
                "target_d5_move",
                data=buffer["target_d5_move"],
                maxshape=(None,),
                dtype="int16",
            )
            f.create_dataset(
                "target_d5_score",
                data=buffer["target_d5_score"],
                maxshape=(None,),
                dtype="float16",
            )

        else:
            # Resize and append
            tensor_data = np.array(buffer["input_tensor"], dtype=np.float32)
            tensor_data[:, 18, :, :] *= 100
            tensor_data = np.round(tensor_data).astype(np.uint8)
            tensor_data = np.transpose(tensor_data, (0, 2, 3, 1))

            for key in [
                "input_tensor",
                "target_d1_move",
                "target_d1_score",
                "target_d5_move",
                "target_d5_score",
            ]:
                dset = f[key]
                if key == "input_tensor":
                    dset.resize(dset.shape[0] + n_samples, axis=0)
                    dset[-n_samples:] = tensor_data
                else:
                    dset.resize(dset.shape[0] + n_samples, axis=0)
                    dset[-n_samples:] = buffer[key]


def clear_buffer(buffer):
    for k in buffer:
        buffer[k] = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn_file", help="Path to PGN file")
    parser.add_argument("output_file", help="Path to output H5 file")
    parser.add_argument("--max_games", type=int, default=None)
    args = parser.parse_args()

    build_dataset(args.pgn_file, args.output_file, args.max_games)
