import os
import re
import subprocess

import h5py
import numpy as np

from utile import initialze_board, BOARD_SIZE   # ⚠️ on n'importe PLUS has_tile_to_flip

# ================== CONFIG ==================

MODELS = [
    r"C:\Users\pc\Desktop\omar\deeplearning2playothello\save_models_CNN\model_28.pt",
    r"C:\Users\pc\Desktop\omar\deeplearning2playothello\save_models_CNN\model_28.pt",
    # tu peux ajouter d'autres modèles ici
    # r"...\model_20.pt",
]

GAME_SCRIPT = r"C:\Users\pc\Desktop\omar\deeplearning2playothello\game.py"

N_GAMES_PER_PAIR = 1

OUTPUT_DIR = "selfplay_h5"
MAX_TURNS = 60

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== LOGIQUE OTHELLO (SANS has_tile_to_flip) ==================

# 8 directions (vertical, horizontal, diagonales)
DIRECTIONS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]


def apply_move(board: np.ndarray, x: int, y: int, player: int) -> np.ndarray:
    """
    Applique un coup Othello standard pour player = 1 (noir) ou -1 (blanc)
    en retournant les pions capturés dans les 8 directions.
    On ne dépend PLUS de has_tile_to_flip ici.
    """
    new_board = board.copy()

    # case déjà occupée -> on ne fait rien (sécurité)
    if new_board[x, y] != 0:
        return new_board

    flipped_any = False

    for dx, dy in DIRECTIONS:
        cx, cy = x + dx, y + dy
        path = []

        # on avance tant qu'on voit des pions adverses
        while 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE and new_board[cx, cy] == -player:
            path.append((cx, cy))
            cx += dx
            cy += dy

        # si on termine sur un pion à nous et qu'il y a au moins un pion adverse entre les deux
        if 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE and new_board[cx, cy] == player and len(path) > 0:
            flipped_any = True
            for fx, fy in path:
                new_board[fx, fy] = player

    if flipped_any:
        new_board[x, y] = player

    return new_board


def moves_log_to_board_sequence(moves_str: str) -> np.ndarray:
    """
    Convertit une string de coups (ex: '56644346...' avec '_' pour pass)
    en une séquence de plateaux (MAX_TURNS, 8, 8) avec -1 / 0 / 1.
    """
    board = initialze_board()
    boards = np.zeros((MAX_TURNS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    player = 1  # noir commence
    t = 0

    i = 0
    L = len(moves_str)

    while i < L and t < MAX_TURNS:
        c = moves_str[i]

        # pass
        if c == "_":
            player *= -1
            i += 1
            continue

        if i + 1 >= L:
            break

        c2 = moves_str[i + 1]
        if c2 == "_":
            player *= -1
            i += 2
            continue

        try:
            x = int(c) - 1
            y = int(c2) - 1
        except ValueError:
            i += 2
            player *= -1
            continue

        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            i += 2
            player *= -1
            continue

        board = apply_move(board, x, y, player)
        boards[t] = board
        t += 1

        player *= -1
        i += 2

    return boards


# ================== OUTILS I/O ==================


def play_match_and_get_logs(model_black: str, model_white: str):
    cmd = ["python", GAME_SCRIPT, model_black, model_white]
    out = subprocess.check_output(cmd, text=True, errors="ignore")
    logs = re.findall(r"Moves log:\s*([0-9_]+)", out)
    return logs, out


def model_short_name(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def get_h5_path_and_dataset(model_a: str, model_b: str):
    name_a = model_short_name(model_a)
    name_b = model_short_name(model_b)
    game_name = f"{name_a}_vs_{name_b}"
    h5_path = os.path.join(OUTPUT_DIR, game_name + ".h5")
    return h5_path, game_name


# ================== GÉNÉRATION & SAUVEGARDE ==================


def generate_selfplay_data():
    for i in range(len(MODELS)):
        for j in range(i, len(MODELS)):   # ici i, len(MODELS) pour autoriser model_28_vs_model_28
            model_a = MODELS[i]
            model_b = MODELS[j]

            h5_path, dataset_name = get_h5_path_and_dataset(model_a, model_b)
            print(f"\n=== Génération pour {dataset_name} → {h5_path} ===")

            all_games = []

            for k in range(N_GAMES_PER_PAIR):
                print(f"  Match {k + 1}/{N_GAMES_PER_PAIR} ...")
                logs, raw_out = play_match_and_get_logs(model_a, model_b)

                if len(logs) == 0:
                    print("    ⚠ Aucun 'Moves log' trouvé, sortie :")
                    print(raw_out)
                    continue

                for lg in logs:
                    boards = moves_log_to_board_sequence(lg)  # (MAX_TURNS, 8, 8)
                    all_games.append(boards)

            if not all_games:
                print(f"  ⚠ Aucun jeu enregistré pour {dataset_name}, fichier ignoré.")
                continue

            data = np.stack(all_games, axis=0)  # (N_games, 60, 8, 8)
            print(f"  Nombre de parties : {data.shape[0]}, shape = {data.shape}")

            with h5py.File(h5_path, "w") as h5f:
                h5f.create_dataset(dataset_name, data=data, dtype="float32")

            print(f"  ✅ Sauvegardé dans {h5_path} (dataset='{dataset_name}')")


if __name__ == "__main__":
    generate_selfplay_data()
