import chess
import chess.engine


class EngineHandler:
    def __init__(
        self, stockfish_path="/usr/games/stockfish", threads=4, hash_size=1024
    ):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({"Threads": threads, "Hash": hash_size})

    def analyze_position(self, fen, depths=[1, 2, 3, 4]):
        board = chess.Board(fen)
        results = {}

        for depth in depths:
            # MultiPV=3 as recommended in specs for policy distribution
            info = self.engine.analyse(
                board, chess.engine.Limit(depth=depth), multipv=3
            )

            # Extract info from the best move (first pv)
            best_info = info[0]

            score_cp = best_info["score"].white().score(mate_score=1500)
            if score_cp is not None:
                # Clipping to [-1500, 1500]
                score_cp = max(-1500, min(1500, score_cp))
            else:
                score_cp = 0  # Fallback

            best_move = (
                best_info["pv"][0].uci()
                if "pv" in best_info and best_info["pv"]
                else None
            )

            # Policy distribution (simplistic approach: just list top moves and their scores)
            policy = []
            for pv in info:
                move = pv["pv"][0].uci() if "pv" in pv and pv["pv"] else None
                s = pv["score"].white().score(mate_score=1500)
                if s is not None:
                    s = max(-1500, min(1500, s))
                policy.append({"move": move, "score": s})

            results[depth] = {
                "score": score_cp,
                "best_move": best_move,
                "policy": policy,
            }

        return results

    def close(self):
        self.engine.quit()

    def __del__(self):
        try:
            self.engine.quit()
        except Exception:
            pass
