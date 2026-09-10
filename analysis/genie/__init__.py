"""GENIE BPC Breast Cancer adapter.

Kept separate from ``analysis/dynamic`` on purpose: the simulator's schema is
dataset-neutral by design (see CLAUDE.md), so every dataset-specific assumption
lives in its own adapter package. METABRIC's lives in ``analysis/01~02``;
K-CURE's will live in ``analysis/kcure``.
"""
