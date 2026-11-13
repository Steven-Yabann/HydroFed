"""Which plants reach the server in each federated round.

Remote plants may lose their connection, so a round can only average the
plants that report in. A schedule lists, for every round, the plants that
take part. Plants that miss rounds still receive the final shared model when
they reconnect.
"""

from __future__ import annotations

import numpy as np

AFFECTED_PLANTS = ("plant_4", "plant_5")
SAMPLED_PER_ROUND = (4, 3, 2)
LATE_JOIN_ROUND = 15


def full_schedule(plants: list[str], rounds: int) -> list[list[str]]:
    """Every plant takes part in every round."""
    return [list(plants) for _ in range(rounds)]


def sampled_schedule(plants: list[str], rounds: int, per_round: int, seed: int) -> list[list[str]]:
    """Sample ``per_round`` plants uniformly each round, as Flower's fraction_fit does."""
    if not 1 <= per_round <= len(plants):
        raise ValueError("per_round must be between 1 and the number of plants")
    rng = np.random.default_rng(seed)
    return [
        [plants[index] for index in sorted(rng.choice(len(plants), per_round, replace=False))]
        for _ in range(rounds)
    ]


def outage_schedule(
    plants: list[str],
    rounds: int,
    offline: tuple[str, ...],
    first_online_round: int,
) -> list[list[str]]:
    """Keep ``offline`` plants out of every round before ``first_online_round``.

    Rounds are counted from 0. A ``first_online_round`` equal to ``rounds``
    keeps those plants out of training altogether.
    """
    return [
        [plant for plant in plants if plant not in offline or round_index >= first_online_round]
        for round_index in range(rounds)
    ]


def connectivity_scenarios(plants: list[str], rounds: int, seed: int) -> dict[str, list[list[str]]]:
    """Return the fixed connectivity scenarios for one training seed."""
    return {
        "full": full_schedule(plants, rounds),
        **{
            f"sampled_{count}_of_{len(plants)}": sampled_schedule(plants, rounds, count, seed)
            for count in SAMPLED_PER_ROUND
        },
        "late_join": outage_schedule(plants, rounds, AFFECTED_PLANTS, LATE_JOIN_ROUND),
        "never_join": outage_schedule(plants, rounds, AFFECTED_PLANTS, rounds),
    }
