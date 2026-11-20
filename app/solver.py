"""Integer calibration solver.

The model is a transportation problem on a bipartite graph:

* left nodes  = streets
* right nodes = housing types
* each household is an edge (street -> housing type) carrying its weight

Every weight is an integer in ``[lower, upper]``.  Unlocked households use
``[1, 2 * initial_weight]``; a locked household is fixed to its locked value.
Constraints are exact sums on some streets and some housing types.  Because
the constraint matrix is a bipartite incidence matrix (totally unimodular),
integer feasibility can be decided with a lower-bounded circulation / max-flow
computation; every feasible flow yields an integral weight vector.

The same routine is also used to test arbitrary subsets of the target and lock
constraints, which is what makes the irreducible-conflict (IIS) search cheap.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

Constraint = Tuple[str, str]  # ("street" | "type" | "lock", group name)


@dataclass(frozen=True)
class Household:
    id: int
    sample_id: str
    street: str
    housing_type: str
    initial_weight: int
    current_weight: int


@dataclass
class Solution:
    feasible: bool
    weights: Optional[Dict[int, int]] = None


class _Edge:
    __slots__ = ("to", "rev", "cap", "initial")

    def __init__(self, to: int, rev: int, cap: int) -> None:
        self.to = to
        self.rev = rev
        self.cap = cap
        self.initial = cap


class _Dinic:
    def __init__(self, node_count: int) -> None:
        self.graph: List[List[_Edge]] = [[] for _ in range(node_count)]

    def add_edge(self, u: int, v: int, cap: int) -> Tuple[int, int]:
        """Add an edge and return a (node, index) reference to its residual edge."""
        forward = _Edge(v, len(self.graph[v]), cap)
        backward = _Edge(u, len(self.graph[u]), 0)
        self.graph[u].append(forward)
        self.graph[v].append(backward)
        return (u, len(self.graph[u]) - 1)

    def flow_on(self, ref: Tuple[int, int]) -> int:
        node, index = ref
        edge = self.graph[node][index]
        return edge.initial - edge.cap

    def max_flow(self, source: int, sink: int) -> int:
        total = 0
        node_count = len(self.graph)
        while True:
            level = [-1] * node_count
            level[source] = 0
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for edge in self.graph[node]:
                    if edge.cap > 0 and level[edge.to] < 0:
                        level[edge.to] = level[node] + 1
                        queue.append(edge.to)
            if level[sink] < 0:
                return total

            pointer = [0] * node_count

            def push(node: int, pushed: int) -> int:
                if node == sink:
                    return pushed
                while pointer[node] < len(self.graph[node]):
                    edge = self.graph[node][pointer[node]]
                    if edge.cap > 0 and level[edge.to] == level[node] + 1:
                        sent = push(edge.to, min(pushed, edge.cap))
                        if sent:
                            edge.cap -= sent
                            self.graph[edge.to][edge.rev].cap += sent
                            return sent
                    pointer[node] += 1
                return 0

            while True:
                sent = push(source, 1 << 120)
                if not sent:
                    break
                total += sent


def _bounds(
    household: Household,
    locked_streets: Set[str],
    lock_weights: Mapping[int, int],
) -> Tuple[int, int]:
    if household.street in locked_streets:
        if household.id not in lock_weights:
            raise ValueError(
                f"街道 {household.street!r} 已锁定，但样本 {household.sample_id!r} 缺少锁定权重"
            )
        value = lock_weights[household.id]
        if value <= 0:
            raise ValueError("锁定权重必须为正整数")
        return value, value
    return 1, 2 * household.initial_weight


def _solve(
    households: Sequence[Household],
    street_targets: Mapping[str, int],
    type_targets: Mapping[str, int],
    lock_weights: Mapping[int, int],
    locked_streets: Set[str],
) -> Solution:
    """Solve the (possibly partial) constraint system.

    A group name absent from a target mapping is unconstrained.
    """
    if any(value < 0 for value in street_targets.values()) or any(
        value < 0 for value in type_targets.values()
    ):
        return Solution(False)

    street_names = sorted({h.street for h in households} | set(street_targets))
    type_names = sorted({h.housing_type for h in households} | set(type_targets))
    street_index = {name: 1 + i for i, name in enumerate(street_names)}
    type_offset = 1 + len(street_names)
    type_index = {name: type_offset + i for i, name in enumerate(type_names)}

    reservoir = 0
    super_source = type_offset + len(type_names)
    super_sink = super_source + 1
    network = _Dinic(super_sink + 1)

    # balance[v] = inflow from lower bounds - outflow from lower bounds.
    balance = [0] * (super_sink + 1)

    def add_bounded(u: int, v: int, lower: int, upper: int) -> Optional[Tuple[int, int]]:
        if lower < 0 or lower > upper:
            return None
        reference = network.add_edge(u, v, upper - lower)
        balance[u] -= lower
        balance[v] += lower
        return reference

    bounds: Dict[int, Tuple[int, int]] = {}
    upper_total = 0
    for household in households:
        lower, upper = _bounds(household, locked_streets, lock_weights)
        bounds[household.id] = (lower, upper)
        upper_total += upper

    total_targets = sum(street_targets.values()) + sum(type_targets.values())
    infinity = total_targets + upper_total + 1

    # Street supply (reservoir -> street).  A constrained street receives
    # exactly its target; an unconstrained street may take anything.
    for street in street_names:
        if street in street_targets:
            target = street_targets[street]
            if add_bounded(reservoir, street_index[street], target, target) is None:
                return Solution(False)
        else:
            add_bounded(reservoir, street_index[street], 0, infinity)

    # Housing-type demand (type -> reservoir).
    for housing_type in type_names:
        node = type_index[housing_type]
        if housing_type in type_targets:
            target = type_targets[housing_type]
            if add_bounded(node, reservoir, target, target) is None:
                return Solution(False)
        else:
            add_bounded(node, reservoir, 0, infinity)

    household_refs: Dict[int, Tuple[int, int]] = {}
    household_lower: Dict[int, int] = {}
    for household in households:
        lower, upper = bounds[household.id]
        reference = add_bounded(
            street_index[household.street],
            type_index[household.housing_type],
            lower,
            upper,
        )
        if reference is None:
            return Solution(False)
        household_refs[household.id] = reference
        household_lower[household.id] = lower

    required = 0
    for node, value in enumerate(balance):
        if value > 0:
            network.add_edge(super_source, node, value)
            required += value
        elif value < 0:
            network.add_edge(node, super_sink, -value)

    if network.max_flow(super_source, super_sink) != required:
        return Solution(False)

    weights: Dict[int, int] = {}
    for household in households:
        weights[household.id] = household_lower[household.id] + network.flow_on(
            household_refs[household.id]
        )
    return Solution(True, weights)


def solve(
    households: Sequence[Household],
    street_targets: Mapping[str, int],
    type_targets: Mapping[str, int],
    lock_weights: Optional[Mapping[int, int]] = None,
    locked_streets: Optional[Iterable[str]] = None,
) -> Solution:
    """Compute one integral weight vector satisfying all given constraints."""
    return _solve(
        households,
        street_targets,
        type_targets,
        dict(lock_weights or {}),
        set(locked_streets or ()),
    )


def _subset_constraints(
    constraints: Set[Constraint],
    street_targets: Mapping[str, int],
    type_targets: Mapping[str, int],
    locked_streets: Set[str],
) -> Tuple[Dict[str, int], Dict[str, int], Set[str]]:
    return (
        {name: value for name, value in street_targets.items() if ("street", name) in constraints},
        {name: value for name, value in type_targets.items() if ("type", name) in constraints},
        {name for name in locked_streets if ("lock", name) in constraints},
    )


def _subset_feasible(
    households: Sequence[Household],
    constraints: Set[Constraint],
    street_targets: Mapping[str, int],
    type_targets: Mapping[str, int],
    lock_weights: Mapping[int, int],
    locked_streets: Set[str],
) -> bool:
    subset_streets, subset_types, subset_locks = _subset_constraints(
        constraints, street_targets, type_targets, locked_streets
    )
    return _solve(
        households, subset_streets, subset_types, lock_weights, subset_locks
    ).feasible


def _deletion_filter(
    order: Sequence[Constraint],
    households: Sequence[Household],
    street_targets: Mapping[str, int],
    type_targets: Mapping[str, int],
    lock_weights: Mapping[int, int],
    locked_streets: Set[str],
) -> List[Constraint]:
    """Greedy deletion pass over ``order``; the whole set must be infeasible.

    Dropping a constraint can only relax the system, so once removing ``c``
    leaves a feasible system, it stays feasible after any further removals.
    One pass therefore yields an irreducible infeasible subset.
    """
    current = list(order)
    for constraint in list(order):
        trial = set(current) - {constraint}
        if not _subset_feasible(
            households, trial, street_targets, type_targets, lock_weights, locked_streets
        ):
            current = [item for item in current if item != constraint]
    return current


def find_conflict(
    households: Sequence[Household],
    street_targets: Mapping[str, int],
    type_targets: Mapping[str, int],
    lock_weights: Optional[Mapping[int, int]] = None,
    locked_streets: Optional[Iterable[str]] = None,
) -> Optional[List[Constraint]]:
    """Return an irreducible infeasible subset, or ``None`` if feasible.

    The returned list contains target conditions and lock conditions.  Every
    proper subset obtained by deleting one of its members is feasible, and the
    full set is not.  Base bounds (positive weight, at most 2x initial) are not
    removable and are not part of the result.
    """
    locks_by_household = dict(lock_weights or {})
    locked = set(locked_streets or ())
    all_constraints: List[Constraint] = [
        ("street", name) for name in sorted(street_targets)
    ] + [("type", name) for name in sorted(type_targets)] + [
        ("lock", name) for name in sorted(locked)
    ]
    full = set(all_constraints)
    if _subset_feasible(
        households, full, street_targets, type_targets, locks_by_household, locked
    ):
        return None

    current = list(all_constraints)
    orders = [
        list(all_constraints),
        list(reversed(all_constraints)),
        [("lock", name) for name in sorted(locked)]
        + [("street", name) for name in sorted(street_targets)]
        + [("type", name) for name in sorted(type_targets)],
        [("type", name) for name in sorted(type_targets)]
        + [("street", name) for name in sorted(street_targets)]
        + [("lock", name) for name in sorted(locked)],
    ]
    for order in orders:
        candidate = _deletion_filter(
            order,
            households,
            street_targets,
            type_targets,
            locks_by_household,
            locked,
        )
        if len(candidate) < len(current):
            current = candidate

    # Defensive verification: the result must be infeasible and every
    # one-item deletion must be feasible.
    final = set(current)
    if _subset_feasible(
        households, final, street_targets, type_targets, locks_by_household, locked
    ):
        raise AssertionError("内部错误：冲突集合意外可行")
    for constraint in final:
        if not _subset_feasible(
            households,
            final - {constraint},
            street_targets,
            type_targets,
            locks_by_household,
            locked,
        ):
            raise AssertionError("内部错误：冲突集合不是极小冲突集")
    return sorted(final)
