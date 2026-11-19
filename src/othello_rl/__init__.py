"""Self-play DQN for Othello (CS 401R final project). Plan: docs/spec.md.

Module map (spec §4.1):
    engine     rules on bitboards; no torch
    players    random / greedy / positional / network players
    solver     4×4 full solve and π*; 6×6 endgame WLD solver
    nets       Q-network and checkpoint I/O
    replay     transitions, replay buffer, symmetry augmentation
    targets    Double-DQN targets; the `target` factor (env | negamax)
    selfplay   batched games and transition cutting; the `target` factor
    pool       opponent sampling; the `sampling` factor (naive | pool)
    train      training loop (python -m othello_rl.train)
    evaluate   opening banks, matches, metrics, cross-play
    regret     MADRID-style checkpoint regret
    export     ONNX + web demo data + parity fixtures
    config     experiment config (dataclass < YAML < CLI)
    rundir     run directories, metadata, checkpoint paths, JSONL logs
"""

__version__ = "0.1.0"
