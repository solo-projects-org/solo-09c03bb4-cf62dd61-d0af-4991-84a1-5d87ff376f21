"""Solver unit tests: feasibility, bounds, locking, and conflict minimality."""

from __future__ import annotations

import itertools
import random
from typing import Dict, Sequence, Set

import pytest

from app import solver


def household(
    hid: int, street: str, housing_type: str, initial: int
) -> solver.Household:
    return solver.Household(hid, f"S{hid:03d}", street, housing_type, initial, initial)


def brute_force(
    households: Sequence[solver.Household],
    street_targets: Dict[str, int],
    type_targets: Dict[str, int],
    lock_weights: Dict[int, int],
    locked_streets: Set[str],
) -> bool:
    ranges = []
    for item in households:
        if item.street in locked_streets:
            ranges.append(range(lock_weights[item.id], lock_weights[item.id] + 1))
        else:
            ranges.append(range(1, 2 * item.initial_weight + 1))
    for combo in itertools.product(*ranges):
        weights = {item.id: value for item, value in zip(households, combo)}
        street_ok = all(
            sum(weights[h.id] for h in households if h.street == name) == value
            for name, value in street_targets.items()
        )
        type_ok = all(
            sum(weights[h.id] for h in households if h.housing_type == name) == value
            for name, value in type_targets.items()
        )
        if street_ok and type_ok:
            return True
    return False


def test_exact_targets_and_bounds():
    households = [
        household(1, "A", "T1", 3),
        household(2, "A", "T2", 5),
        household(3, "B", "T1", 2),
        household(4, "B", "T2", 4),
    ]
    result = solver.solve(
        households, {"A": 9, "B": 7}, {"T1": 8, "T2": 8}
    )
    assert result.feasible
    weights = result.weights
    assert sum(weights[h.id] for h in households if h.street == "A") == 9
    assert sum(weights[h.id] for h in households if h.street == "B") == 7
    assert sum(weights[h.id] for h in households if h.housing_type == "T1") == 8
    assert sum(weights[h.id] for h in households if h.housing_type == "T2") == 8
    for item in households:
        assert 1 <= weights[item.id] <= 2 * item.initial_weight
        assert isinstance(weights[item.id], int)


def test_infeasible_total_mismatch():
    households = [household(1, "A", "T1", 2)]
    assert not solver.solve(households, {"A": 5}, {"T1": 4}).feasible


def test_doubling_bound_is_enforced():
    households = [household(1, "A", "T1", 2)]
    assert solver.solve(households, {"A": 4}, {"T1": 4}).feasible
    assert not solver.solve(households, {"A": 5}, {"T1": 5}).feasible


def test_unconstrained_groups_are_free():
    households = [household(1, "A", "T1", 2), household(2, "B", "T2", 2)]
    result = solver.solve(households, {"A": 3}, {"T2": 4})
    assert result.feasible
    assert result.weights[1] == 3
    assert result.weights[2] == 4


def test_locked_weights_are_preserved():
    households = [household(1, "A", "T1", 3), household(2, "A", "T2", 4),
                  household(3, "B", "T1", 2)]
    result = solver.solve(
        households, {"B": 3}, {"T2": 5}, {1: 2, 2: 5}, {"A"}
    )
    assert result.feasible
    assert result.weights[1] == 2
    assert result.weights[2] == 5
    assert result.weights[3] == 3


def test_lock_can_force_conflict():
    households = [household(1, "A", "T1", 2)]
    # locked at 3, but type target demands 2: impossible.
    assert not solver.solve(households, {}, {"T1": 2}, {1: 3}, {"A"}).feasible


def test_conflict_is_irreducible():
    households = [
        household(1, "A", "T1", 2),
        household(2, "A", "T2", 2),
        household(3, "B", "T1", 2),
        household(4, "B", "T2", 2),
    ]
    street_targets = {"A": 7, "B": 4}
    type_targets = {"T1": 4, "T2": 4}
    conflict = solver.find_conflict(households, street_targets, type_targets)
    assert conflict is not None
    subset = set(conflict)
    assert not solver._subset_feasible(
        households, subset, street_targets, type_targets, {}, set()
    )
    for item in subset:
        assert solver._subset_feasible(
            households, subset - {item}, street_targets, type_targets, {}, set()
        ), f"删除 {item} 后仍无解，冲突集不极小"


def test_conflict_includes_involved_lock():
    households = [household(1, "A", "T1", 2), household(2, "B", "T1", 2)]
    conflict = solver.find_conflict(
        households, {"B": 4}, {"T1": 3}, {1: 3}, {"A"}
    )
    assert conflict is not None
    assert ("lock", "A") in conflict
    assert ("type", "T1") in conflict
    assert ("street", "B") not in conflict


def test_no_conflict_when_feasible():
    households = [household(1, "A", "T1", 2)]
    assert solver.find_conflict(households, {"A": 3}, {"T1": 3}) is None


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_random_feasibility_matches_brute_force(seed: int):
    rng = random.Random(seed)
    for _ in range(120):
        count = rng.randint(2, 5)
        households = [
            household(
                index,
                rng.choice(["A", "B"]),
                rng.choice(["T1", "T2"]),
                rng.randint(1, 3),
            )
            for index in range(count)
        ]
        streets = sorted({item.street for item in households})
        types = sorted({item.housing_type for item in households})
        street_targets = {
            name: rng.randint(0, 8) for name in streets if rng.random() < 0.6
        }
        type_targets = {
            name: rng.randint(0, 8) for name in types if rng.random() < 0.6
        }
        locked: Set[str] = set()
        lock_weights: Dict[int, int] = {}
        if rng.random() < 0.3 and streets:
            locked.add(rng.choice(streets))
            for item in households:
                if item.street in locked:
                    lock_weights[item.id] = rng.randint(1, 2 * item.initial_weight)

        result = solver.solve(
            households, street_targets, type_targets, lock_weights, locked
        )
        expected = brute_force(
            households, street_targets, type_targets, lock_weights, locked
        )
        assert result.feasible == expected
        if result.feasible:
            weights = result.weights
            for name, value in street_targets.items():
                assert sum(weights[h.id] for h in households if h.street == name) == value
            for name, value in type_targets.items():
                assert (
                    sum(weights[h.id] for h in households if h.housing_type == name)
                    == value
                )
            for item in households:
                if item.street in locked:
                    assert weights[item.id] == lock_weights[item.id]
                else:
                    assert 1 <= weights[item.id] <= 2 * item.initial_weight


@pytest.mark.parametrize("seed", [11, 12])
def test_random_conflicts_are_minimal(seed: int):
    rng = random.Random(seed)
    checked = 0
    for _ in range(120):
        count = rng.randint(2, 5)
        households = [
            household(
                index,
                rng.choice(["A", "B"]),
                rng.choice(["T1", "T2"]),
                rng.randint(1, 3),
            )
            for index in range(count)
        ]
        streets = sorted({item.street for item in households})
        types = sorted({item.housing_type for item in households})
        street_targets = {
            name: rng.randint(0, 8) for name in streets if rng.random() < 0.7
        }
        type_targets = {
            name: rng.randint(0, 8) for name in types if rng.random() < 0.7
        }
        locked: Set[str] = set()
        lock_weights: Dict[int, int] = {}
        if rng.random() < 0.3 and streets:
            locked.add(rng.choice(streets))
            for item in households:
                if item.street in locked:
                    lock_weights[item.id] = rng.randint(1, 2 * item.initial_weight)
        if solver.solve(households, street_targets, type_targets, lock_weights, locked).feasible:
            continue
        conflict = solver.find_conflict(
            households, street_targets, type_targets, lock_weights, locked
        )
        assert conflict is not None
        subset = set(conflict)
        assert not solver._subset_feasible(
            households, subset, street_targets, type_targets, lock_weights, locked
        )
        for item in subset:
            assert solver._subset_feasible(
                households,
                subset - {item},
                street_targets,
                type_targets,
                lock_weights,
                locked,
            ), f"删除 {item} 后仍无解"
        checked += 1
    assert checked > 0
