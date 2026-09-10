"""Taking the response channel away from the searching policy (v1.7).

v1.5 decomposed the MCTS-minus-NCCN gap and found that 81% of it came from two
asymmetries we had declared ourselves. What it could **not** separate was the
one advantage a sequential policy is *supposed* to have: adapting to what it
observes. The null control (+0.0013, SE 0.0017) contained that advantage and
could not be told apart from noise.

v1.6 then showed the thing is real in practice - in GENIE BPC, a "Progressing"
scan is followed by a new regimen at 2.41x the rate a "Stable" scan is, and the
placebo-in-time control holds. So the question is no longer whether closed-loop
adaptation exists, but **how much of our simulated advantage is made of it.**

The response reaches a planner by two separate routes:

``label``
    ``state.response`` multiplies the terminal hazard. v0.5 divides the channel
    by its own expectation so choosing neoadjuvant buys nothing on average, but
    the *relative* ordering survives - "major" is still better news than "none".
    A planner that sees the label knows something about its own prognosis.

``tumour size``
    The response shrinks the tumour, and ``bcs_max_tumor_mm`` gates breast-
    conserving surgery on the result. This route changes the **legal action
    set**, not just the value estimate.

Three environments, therefore, differing only in what the *planner* is told:

===========  =======================  ============================
mode         label the planner sees   size the planner sees
===========  =======================  ============================
``none``     the drawn response       the realised size
``label``    ``not_applicable``       the realised size
``full``     ``not_applicable``       E[size] under the chosen intensity
===========  =======================  ============================

``not_applicable`` is the right stand-in for "I do not know": the environment
skips the response block entirely for it, giving a hazard factor of exactly
1.0, and the neutralised channel has expectation exactly 1.0 by construction.

The policy plans in the blinded environment and **acts in the real one**, which
is what "cannot use the response channel" means operationally. NCCN never reads
``state.response`` at all, so its utility is unchanged across all three modes -
an equality the run script asserts rather than hopes for.
"""

from __future__ import annotations

from dataclasses import replace
import math
import random

from .config import DynamicConfig
from .environment import Action, DynamicBreastCancerEnvironment
from .policies import CachedMCTSPolicy
from .schema import DynamicState, PatientProfile

#: What the planner is allowed to see. See the module docstring.
BLINDING_MODES = ("none", "label", "full")

#: The response label that means "nothing was observed". The environment gives
#: it a hazard factor of exactly 1.0 by skipping the response block.
UNOBSERVED = "not_applicable"


def expected_post_chemo_size(
    pre_size: float,
    intensity: str,
    config: DynamicConfig,
) -> float:
    """Tumour size a blinded planner should expect after neoadjuvant chemo."""
    return round(pre_size * config.tumor_multiplier_mean(intensity), 2)


def blind_state(
    state: DynamicState,
    mode: str,
    patient: PatientProfile,
    config: DynamicConfig,
) -> DynamicState:
    """Map an observed state onto the information state the planner may use.

    Identity when nothing was observed, so a surgery-first patient - who never
    draws a response - is planned for identically in all three modes.
    """
    if mode not in BLINDING_MODES:
        raise ValueError(f"unknown blinding mode: {mode!r}")
    if mode == "none" or state.response == UNOBSERVED:
        return state
    if mode == "label":
        return replace(state, response=UNOBSERVED)
    if state.chemo is None:
        raise ValueError("a response was drawn but no chemotherapy is recorded")
    return replace(
        state,
        response=UNOBSERVED,
        current_tumor_size_mm=expected_post_chemo_size(
            round(patient.tumor_size_mm, 2), state.chemo, config),
    )


class ResponseBlindEnvironment(DynamicBreastCancerEnvironment):
    """A copy of the environment with the response channel hidden or removed.

    Used only for *planning*. Episodes are always simulated in the real
    environment, so every arm faces the same draws and the same outcomes.
    """

    def __init__(
        self,
        environment: DynamicBreastCancerEnvironment,
        mode: str,
    ) -> None:
        if mode not in BLINDING_MODES:
            raise ValueError(f"unknown blinding mode: {mode!r}")
        super().__init__(environment.patient, environment.risk_table,
                         environment.config)
        self.mode = mode

    def _chemo_response(
        self,
        state: DynamicState,
        action: Action,
        rng: random.Random,
    ) -> tuple[str, float]:
        response, size = super()._chemo_response(state, action, rng)
        if self.mode == "none":
            return response, size
        if self.mode == "label":
            return UNOBSERVED, size
        # ``full``: the draw above is discarded rather than skipped, so the
        # rollout RNG stream stays aligned with the other two modes. Common
        # random numbers, not laziness - it removes a source of between-arm
        # variance that has nothing to do with blinding.
        return UNOBSERVED, expected_post_chemo_size(
            state.current_tumor_size_mm, action, self.config)


class ResponseBlindMCTSPolicy:
    """Plan without the response channel; act in the real environment.

    Decisions are cached on the *information* state, so every realisation that
    a blinded planner cannot tell apart gets the same action - which is what
    being unable to observe something means.
    """

    def __init__(
        self,
        environment: DynamicBreastCancerEnvironment,
        mode: str = "full",
        simulations: int = 128,
        exploration_weight: float = math.sqrt(2.0),
        seed: int = 0,
    ) -> None:
        if mode not in BLINDING_MODES:
            raise ValueError(f"unknown blinding mode: {mode!r}")
        self.environment = environment
        self.mode = mode
        self.planner_environment = ResponseBlindEnvironment(environment, mode)
        self.inner = CachedMCTSPolicy(
            self.planner_environment, simulations=simulations,
            exploration_weight=exploration_weight, seed=seed)
        #: Decisions taken from a state the planner was not shown in full.
        self.blinded_decisions = 0
        #: Times the blinded plan turned out to be illegal once the realised
        #: tumour size was applied.
        self.fallbacks = 0

    @property
    def cache(self):  # pragma: no cover - convenience for parity with the base
        return self.inner.cache

    def __call__(self, state: DynamicState) -> str:
        legal = self.environment.legal_actions(state)
        if len(legal) == 1:
            return legal[0]
        information = blind_state(
            state, self.mode, self.environment.patient, self.environment.config)
        if information != state:
            self.blinded_decisions += 1
        action = self.inner(information)
        if action in legal:
            return action
        # Tumour size gates BCS and nothing else, so when a blinded plan is
        # illegal the only survivor is mastectomy. Anything else means a second
        # size-gated action appeared and this fallback rule is no longer safe.
        if len(legal) != 1:
            raise RuntimeError(
                f"blinded action {action!r} is illegal in {state!r} and the "
                f"fallback is ambiguous among {legal!r}")
        self.fallbacks += 1
        return legal[0]
