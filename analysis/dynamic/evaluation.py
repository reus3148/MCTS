"""Episode simulation helpers for dynamic policy evaluation."""

from __future__ import annotations

import random
from typing import Callable

from .environment import DynamicBreastCancerEnvironment
from .schema import DynamicState

Policy = Callable[[DynamicState], str]


def simulate_episode(
    environment: DynamicBreastCancerEnvironment,
    policy: Policy,
    seed: int,
    include_trace: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rng = random.Random(seed)
    state = environment.initial_state()
    total_reward = 0.0
    trace: list[dict[str, object]] = []
    steps = 0

    while not environment.is_terminal(state):
        if steps >= 32:
            raise RuntimeError("episode exceeded the dynamic environment horizon")
        phase = state.phase
        action = policy(state)
        next_state, reward, info = environment.step(state, action, rng)
        total_reward += reward
        if include_trace:
            trace.append({
                "step": steps + 1,
                "phase": phase,
                "action": action,
                "reward": reward,
                "event": info.get("event"),
                "response": info.get("response"),
                "toxicity": info.get("toxicity"),
                "tumor_size_mm": next_state.current_tumor_size_mm,
                "year": next_state.year,
                "alive": int(next_state.alive),
                "recurred": int(next_state.recurred),
            })
        state = next_state
        steps += 1

    result = {
        "utility": total_reward,
        "survived_5y": int(state.alive and state.year >= environment.config.horizon_years),
        "recurred_by_5y": int(state.recurred),
        "toxicity_count": state.toxicity_count,
        "timing": state.timing,
        "surgery": state.surgery,
        "chemo": state.chemo,
        "endocrine": state.endocrine,
        "radiation": state.radiation,
        "response": state.response,
        "salvage": state.salvage,
        "recurrence_year": state.recurrence_year,
        "terminal_year": state.year,
    }
    return result, trace


def run_policy_episodes(
    environment: DynamicBreastCancerEnvironment,
    policy: Policy,
    episodes: int,
    seed: int,
) -> list[dict[str, object]]:
    return [
        simulate_episode(
            environment,
            policy,
            seed=seed + episode,
            include_trace=False,
        )[0]
        for episode in range(episodes)
    ]

