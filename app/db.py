"""SQLite persistence layer for the calibration tool."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set

from . import solver

SCHEMA = """
CREATE TABLE IF NOT EXISTS households (
    id             INTEGER PRIMARY KEY,
    sample_id      TEXT NOT NULL UNIQUE,
    street         TEXT NOT NULL,
    housing_type   TEXT NOT NULL,
    initial_weight INTEGER NOT NULL CHECK (initial_weight > 0),
    current_weight INTEGER NOT NULL CHECK (current_weight > 0),
    row_order      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_households_street ON households (street, row_order);
CREATE INDEX IF NOT EXISTS idx_households_type ON households (housing_type, row_order);

CREATE TABLE IF NOT EXISTS street_locks (
    street          TEXT PRIMARY KEY,
    locked_at       TEXT NOT NULL,
    household_count INTEGER NOT NULL CHECK (household_count > 0),
    total_weight    INTEGER NOT NULL CHECK (total_weight > 0)
);

CREATE TABLE IF NOT EXISTS locked_weights (
    household_id INTEGER PRIMARY KEY REFERENCES households (id) ON DELETE CASCADE,
    street       TEXT NOT NULL,
    weight       INTEGER NOT NULL CHECK (weight > 0)
);
CREATE INDEX IF NOT EXISTS idx_locked_weights_street ON locked_weights (street);

CREATE TABLE IF NOT EXISTS publications (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    published_at        TEXT NOT NULL,
    household_count     INTEGER NOT NULL,
    total_weight        INTEGER NOT NULL,
    street_targets_json TEXT NOT NULL,
    type_targets_json   TEXT NOT NULL,
    locked_streets_json TEXT NOT NULL,
    weights_json        TEXT NOT NULL
);
"""

MAX_TARGET = 1_000_000_000

_HEADER_ALIASES = {
    "sample_id": {"样本号", "样本编号", "sample_id", "sample id", "id"},
    "street": {"街道", "街道名称", "street"},
    "housing_type": {"住房类型", "房屋类型", "housing_type", "housing type", "type"},
    "initial_weight": {"初始权重", "初始值", "initial_weight", "initial weight"},
}


class ValidationError(ValueError):
    """Raised when user input or the sample file is invalid."""


@dataclass
class LockOutcome:
    street: str
    count: int
    total: int
    already: bool


@dataclass
class PublishOutcome:
    ok: bool
    updated: int = 0
    total: int = 0
    publication_id: Optional[int] = None
    conflict: Optional[List[dict]] = None
    # Set when a street that was editable in the submitted form has been
    # locked since with a different total; the publish is then rejected.
    lock_conflicts: Optional[List[dict]] = None


@dataclass
class DashboardData:
    households: List[dict] = field(default_factory=list)
    streets: List[dict] = field(default_factory=list)
    types: List[dict] = field(default_factory=list)
    locks: Dict[str, dict] = field(default_factory=dict)
    publications: List[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Sample import
# ---------------------------------------------------------------------------


def _canonical_headers(fieldnames: Optional[Iterable[str]]) -> Dict[str, str]:
    if not fieldnames:
        raise ValidationError("样本 CSV 没有表头")
    mapping: Dict[str, str] = {}
    for raw in fieldnames:
        name = (raw or "").strip()
        for canonical, aliases in _HEADER_ALIASES.items():
            if name in aliases:
                if canonical in mapping:
                    raise ValidationError(f"样本 CSV 表头重复：{name}")
                mapping[canonical] = raw
                break
    missing = [c for c in _HEADER_ALIASES if c not in mapping]
    if missing:
        readable = {
            "sample_id": "样本号",
            "street": "街道",
            "housing_type": "住房类型",
            "initial_weight": "初始权重",
        }
        raise ValidationError(
            "样本 CSV 缺少列：" + "、".join(readable[item] for item in missing)
        )
    return mapping


def parse_sample_csv(path: Path) -> List[dict]:
    if not path.is_file():
        raise ValidationError(f"找不到样本 CSV：{path}")
    rows: List[dict] = []
    seen_ids: Set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = _canonical_headers(reader.fieldnames)
        for line_no, raw in enumerate(reader, start=2):
            sample_id = (raw.get(headers["sample_id"]) or "").strip()
            street = (raw.get(headers["street"]) or "").strip()
            housing_type = (raw.get(headers["housing_type"]) or "").strip()
            weight_text = (raw.get(headers["initial_weight"]) or "").strip()
            if not sample_id:
                raise ValidationError(f"样本 CSV 第 {line_no} 行：样本号为空")
            if sample_id in seen_ids:
                raise ValidationError(f"样本 CSV 第 {line_no} 行：样本号重复（{sample_id}）")
            if not street:
                raise ValidationError(f"样本 CSV 第 {line_no} 行：街道为空")
            if not housing_type:
                raise ValidationError(f"样本 CSV 第 {line_no} 行：住房类型为空")
            if not re.fullmatch(r"[1-9][0-9]*", weight_text):
                raise ValidationError(
                    f"样本 CSV 第 {line_no} 行：初始权重必须是正整数（当前为 {weight_text!r}）"
                )
            seen_ids.add(sample_id)
            rows.append(
                {
                    "sample_id": sample_id,
                    "street": street,
                    "housing_type": housing_type,
                    "initial_weight": int(weight_text),
                }
            )
    if not rows:
        raise ValidationError("样本 CSV 没有任何数据行")
    if len(rows) > 200_000:
        raise ValidationError("样本 CSV 行数过多（上限 200000 行）")
    return rows


def initialize(db_path: Path, sample_csv_path: Path) -> None:
    """Create the schema and seed the fixed sample once."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        count = conn.execute("SELECT COUNT(*) FROM households").fetchone()[0]
        if count:
            return
        rows = parse_sample_csv(Path(sample_csv_path))
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Re-check under the write lock in case another process seeded first.
            if conn.execute("SELECT COUNT(*) FROM households").fetchone()[0]:
                conn.execute("COMMIT")
                return
            conn.executemany(
                """
                INSERT INTO households
                    (sample_id, street, housing_type, initial_weight, current_weight, row_order)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["sample_id"],
                        row["street"],
                        row["housing_type"],
                        row["initial_weight"],
                        row["initial_weight"],
                        index,
                    )
                    for index, row in enumerate(rows)
                ],
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def _load_locks(conn: sqlite3.Connection) -> Dict[str, dict]:
    locks: Dict[str, dict] = {}
    for row in conn.execute("SELECT * FROM street_locks ORDER BY street"):
        locks[row["street"]] = {
            "street": row["street"],
            "locked_at": row["locked_at"],
            "household_count": row["household_count"],
            "total_weight": row["total_weight"],
            "weights": {},
        }
    for row in conn.execute("SELECT household_id, street, weight FROM locked_weights"):
        if row["street"] in locks:
            locks[row["street"]]["weights"][row["household_id"]] = row["weight"]
    return locks


def _households(conn: sqlite3.Connection) -> List[dict]:
    return [
        dict(row)
        for row in conn.execute(
            """
            SELECT id, sample_id, street, housing_type, initial_weight, current_weight
            FROM households ORDER BY row_order
            """
        )
    ]


def get_dashboard(db_path: Path) -> DashboardData:
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN")
        households = _households(conn)
        locks = _load_locks(conn)
        publication_rows = [
            {
                "id": row["id"],
                "published_at": row["published_at"],
                "household_count": row["household_count"],
                "total_weight": row["total_weight"],
            }
            for row in conn.execute(
                "SELECT id, published_at, household_count, total_weight "
                "FROM publications ORDER BY id DESC LIMIT 5"
            )
        ]
        publication_count = conn.execute("SELECT COUNT(*) FROM publications").fetchone()[0]
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    street_groups: Dict[str, dict] = {}
    type_groups: Dict[str, dict] = {}
    for household in households:
        household["locked"] = household["street"] in locks
        street = household["street"]
        if street not in street_groups:
            street_groups[street] = {
                "name": street,
                "households": [],
                "count": 0,
                "initial_sum": 0,
                "current_sum": 0,
                "locked": street in locks,
            }
        group = street_groups[street]
        group["households"].append(household)
        group["count"] += 1
        group["initial_sum"] += household["initial_weight"]
        group["current_sum"] += household["current_weight"]

        housing_type = household["housing_type"]
        if housing_type not in type_groups:
            type_groups[housing_type] = {
                "name": housing_type,
                "households": [],
                "count": 0,
                "initial_sum": 0,
                "current_sum": 0,
            }
        type_group = type_groups[housing_type]
        type_group["households"].append(household)
        type_group["count"] += 1
        type_group["initial_sum"] += household["initial_weight"]
        type_group["current_sum"] += household["current_weight"]

    for name, group in street_groups.items():
        group["lock"] = locks.get(name)
        group["target_value"] = group["current_sum"]
        group["min_target"] = group["count"]
        group["max_target"] = 2 * group["initial_sum"]
        if name in locks:
            group["target_value"] = locks[name]["total_weight"]
            group["min_target"] = locks[name]["total_weight"]
            group["max_target"] = locks[name]["total_weight"]

    for group in type_groups.values():
        group["min_target"] = group["count"]
        group["max_target"] = 2 * group["initial_sum"]

    summary = {
        "household_count": len(households),
        "street_count": len(street_groups),
        "type_count": len(type_groups),
        "total_initial": sum(item["initial_weight"] for item in households),
        "total_current": sum(item["current_weight"] for item in households),
        "locked_streets": len(locks),
        "publication_count": publication_count,
        "last_published_at": publication_rows[0]["published_at"] if publication_rows else None,
    }

    return DashboardData(
        households=households,
        streets=list(street_groups.values()),
        types=list(type_groups.values()),
        locks=locks,
        publications=publication_rows,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------


def lock_street(db_path: Path, street: str) -> LockOutcome:
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            rows = conn.execute(
                "SELECT id, current_weight FROM households WHERE street = ? ORDER BY row_order",
                (street,),
            ).fetchall()
            if not rows:
                raise ValidationError(f"街道不存在：{street}")
            existing = conn.execute(
                "SELECT * FROM street_locks WHERE street = ?", (street,)
            ).fetchone()
            if existing:
                conn.execute("COMMIT")
                return LockOutcome(
                    street=street,
                    count=existing["household_count"],
                    total=existing["total_weight"],
                    already=True,
                )
            total = sum(row["current_weight"] for row in rows)
            conn.execute(
                "INSERT INTO street_locks (street, locked_at, household_count, total_weight) "
                "VALUES (?, ?, ?, ?)",
                (street, _now(), len(rows), total),
            )
            conn.executemany(
                "INSERT INTO locked_weights (household_id, street, weight) VALUES (?, ?, ?)",
                [(row["id"], street, row["current_weight"]) for row in rows],
            )
            conn.execute("COMMIT")
            return LockOutcome(street=street, count=len(rows), total=total, already=False)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


def _solver_households(rows: List[dict]) -> List[solver.Household]:
    return [
        solver.Household(
            id=row["id"],
            sample_id=row["sample_id"],
            street=row["street"],
            housing_type=row["housing_type"],
            initial_weight=row["initial_weight"],
            current_weight=row["current_weight"],
        )
        for row in rows
    ]


def _describe_conflict(
    constraints: List[solver.Constraint],
    street_targets: Mapping[str, int],
    type_targets: Mapping[str, int],
    locks: Mapping[str, dict],
) -> List[dict]:
    described: List[dict] = []
    for kind, name in constraints:
        if kind == "street":
            described.append(
                {
                    "kind": "street",
                    "name": name,
                    "label": f"街道「{name}」目标户数 = {street_targets[name]:,}",
                }
            )
        elif kind == "type":
            described.append(
                {
                    "kind": "type",
                    "name": name,
                    "label": f"住房类型「{name}」目标户数 = {type_targets[name]:,}",
                }
            )
        elif kind == "lock":
            lock = locks[name]
            described.append(
                {
                    "kind": "lock",
                    "name": name,
                    "label": (
                        f"街道「{name}」锁定（{lock['household_count']} 户样本，"
                        f"锁定合计 {lock['total_weight']:,}）"
                    ),
                }
            )
    return described


def publish(
    db_path: Path,
    street_targets: Mapping[str, int],
    type_targets: Mapping[str, int],
    editable_streets: Optional[Iterable[str]] = None,
) -> PublishOutcome:
    """Atomically calibrate and publish; never partially writes weights.

    ``editable_streets`` lists the streets whose target boxes were editable
    when the submitted form was rendered.  If such a street has been locked
    since, its locked total must equal the originally filled target;
    otherwise the publish is rejected and every mismatching street is
    reported in ``lock_conflicts``.  Targets of streets that were already
    read-only in the form keep being ignored in favour of the locked value.
    """
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            rows = _households(conn)
            if not rows:
                raise ValidationError("样本为空，无法校准")
            locks = _load_locks(conn)
            unknown = set(street_targets) - {row["street"] for row in rows}
            if unknown:
                raise ValidationError("目标中出现样本里不存在的街道：" + "、".join(sorted(unknown)))
            for name, value in street_targets.items():
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValidationError(f"街道「{name}」的目标户数必须是非负整数")
                if value > MAX_TARGET:
                    raise ValidationError(f"街道「{name}」的目标户数过大")
            for name, value in type_targets.items():
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValidationError(f"住房类型「{name}」的目标户数必须是非负整数")
                if value > MAX_TARGET:
                    raise ValidationError(f"住房类型「{name}」的目标户数过大")

            # Locks committed before this transaction are the ones the new
            # weights must respect.  A street that was still editable when
            # the form was rendered carries a target the user actually filled
            # in; a lock created since must not silently override it.
            editable = set(editable_streets or ())
            lock_conflicts = [
                {
                    "street": name,
                    "target": street_targets[name],
                    "locked_total": locks[name]["total_weight"],
                    "household_count": locks[name]["household_count"],
                }
                for name in sorted(locks)
                if name in editable
                and name in street_targets
                and street_targets[name] != locks[name]["total_weight"]
            ]
            if lock_conflicts:
                conn.execute("ROLLBACK")
                return PublishOutcome(ok=False, lock_conflicts=lock_conflicts)

            # A locked street's weights are frozen, so its total is already
            # implied by the lock; its submitted value is ignored and only the
            # remaining streets are target conditions the statistician adjusts.
            street_targets = {
                name: value for name, value in street_targets.items() if name not in locks
            }
            streets = {row["street"] for row in rows} - set(locks)
            types = {row["housing_type"] for row in rows}
            if set(street_targets) != streets:
                raise ValidationError("街道目标与样本中的街道不一致")
            if set(type_targets) != types:
                raise ValidationError("住房类型目标与样本中的类型不一致")

            # Only the unlocked streets contribute target constraints; the
            # locked ones contribute their frozen per-household weights.
            lock_weights = {
                household_id: weight
                for lock in locks.values()
                for household_id, weight in lock["weights"].items()
            }
            households = _solver_households(rows)
            locked_streets = set(locks)

            result = solver.solve(
                households, street_targets, type_targets, lock_weights, locked_streets
            )
            if not result.feasible:
                constraints = solver.find_conflict(
                    households, street_targets, type_targets, lock_weights, locked_streets
                )
                conflict = (
                    _describe_conflict(constraints, street_targets, type_targets, locks)
                    if constraints
                    else []
                )
                conn.execute("ROLLBACK")
                return PublishOutcome(ok=False, conflict=conflict)

            weights = result.weights or {}
            for street, lock in locks.items():
                for household_id, locked_weight in lock["weights"].items():
                    if weights.get(household_id) != locked_weight:
                        raise RuntimeError("内部错误：求解结果偏离锁定权重")

            updated = len(weights)
            total = sum(weights.values())
            conn.executemany(
                "UPDATE households SET current_weight = ? WHERE id = ?",
                [(weight, household_id) for household_id, weight in weights.items()],
            )
            conn.execute(
                """
                INSERT INTO publications
                    (published_at, household_count, total_weight, street_targets_json,
                     type_targets_json, locked_streets_json, weights_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now(),
                    updated,
                    total,
                    json.dumps(dict(street_targets), ensure_ascii=False, sort_keys=True),
                    json.dumps(dict(type_targets), ensure_ascii=False, sort_keys=True),
                    json.dumps(sorted(locked_streets), ensure_ascii=False),
                    json.dumps(weights, ensure_ascii=False, sort_keys=True),
                ),
            )
            publication_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute("COMMIT")
            return PublishOutcome(
                ok=True, updated=updated, total=total, publication_id=publication_id
            )
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
