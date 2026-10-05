"""MSK-CHORD (MSK, Nature 2024) adapter.

Kept separate from ``analysis/dynamic`` for the same reason ``analysis/genie``
is (see CLAUDE.md): the simulator's schema is dataset-neutral, so every
dataset-specific assumption lives in its own adapter package.

The release is CC BY-NC-ND 4.0 - **NonCommercial, NoDerivatives**, which is
stricter than GENIE BPC's terms. Neither the raw tables nor derived
patient-level tables go into the repository or the site; reports carry
aggregate counts and per-file SHA-256 only. Cite Jee et al., Nature 2024
(PMID 39506116).

See ``reports/msk-chord-profile-v2.2`` for the eligibility assessment this
package was written to support.
"""
