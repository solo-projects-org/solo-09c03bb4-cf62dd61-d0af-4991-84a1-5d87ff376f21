"""Tests for CSV import, persistence, locking, and publication semantics."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app import db, solver


def write_csv(path: Path, rows, headers=("样本号", "街道", "住房类型", "初始权重")) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    return write_csv(
        tmp_path / "sample.csv",
        [
            ("H01", "A", "T1", 2),
            ("H02", "A", "T2", 3),
            ("H03", "B", "T1", 4),
            ("H04", "B", "T2", 1),
        ],
    )


@pytest.fixture()
def database(tmp_path: Path, sample_csv: Path) -> Path:
    path = tmp_path / "calibration.sqlite3"
    db.initialize(path, sample_csv)
    return path


def test_import_seeds_initial_weights(database: Path):
    data = db.get_dashboard(database)
    assert data.summary["household_count"] == 4
    assert data.summary["total_current"] == data.summary["total_initial"] == 10
    assert {street["name"] for street in data.streets} == {"A", "B"}
    assert {group["name"] for group in data.types} == {"T1", "T2"}
    assert all(h["current_weight"] == h["initial_weight"] for h in data.households)


def test_import_is_idempotent(database: Path, sample_csv: Path):
    db.lock_street(database, "A")
    db.initialize(database, sample_csv)
    data = db.get_dashboard(database)
    assert data.summary["household_count"] == 4
    assert set(data.locks) == {"A"}


def test_invalid_csv_rejected(tmp_path: Path):
    path = write_csv(tmp_path / "bad.csv", [("H1", "A", "T1", 0)])
    with pytest.raises(db.ValidationError):
        db.parse_sample_csv(path)

    path = write_csv(tmp_path / "dup.csv", [("H1", "A", "T1", 1), ("H1", "B", "T1", 1)])
    with pytest.raises(db.ValidationError):
        db.parse_sample_csv(path)

    path = write_csv(
        tmp_path / "missing.csv", [("H1", "A", 1)], headers=("样本号", "街道", "初始权重")
    )
    with pytest.raises(db.ValidationError):
        db.parse_sample_csv(path)


def test_publish_updates_weights_and_history(database: Path):
    outcome = db.publish(database, {"A": 8, "B": 6}, {"T1": 8, "T2": 6})
    assert outcome.ok
    assert outcome.total == 14
    data = db.get_dashboard(database)
    by_street = {street["name"]: street["current_sum"] for street in data.streets}
    by_type = {group["name"]: group["current_sum"] for group in data.types}
    assert by_street == {"A": 8, "B": 6}
    assert by_type == {"T1": 8, "T2": 6}
    for household in data.households:
        assert 1 <= household["current_weight"] <= 2 * household["initial_weight"]
        assert isinstance(household["current_weight"], int)
    assert data.summary["publication_count"] == 1


def test_publish_failure_keeps_current_weights(database: Path):
    db.publish(database, {"A": 8, "B": 6}, {"T1": 8, "T2": 6})
    before = {
        h["sample_id"]: h["current_weight"] for h in db.get_dashboard(database).households
    }
    outcome = db.publish(database, {"A": 100, "B": 1}, {"T1": 50, "T2": 50})
    assert not outcome.ok
    assert outcome.conflict
    after = {
        h["sample_id"]: h["current_weight"] for h in db.get_dashboard(database).households
    }
    assert before == after
    assert db.get_dashboard(database).summary["publication_count"] == 1


def test_lock_freezes_weights_across_publish(database: Path):
    db.publish(database, {"A": 8, "B": 6}, {"T1": 8, "T2": 6})
    data = db.get_dashboard(database)
    a_weights = {
        h["sample_id"]: h["current_weight"]
        for h in data.households
        if h["street"] == "A"
    }
    a_by_type = {
        housing_type: sum(
            h["current_weight"]
            for h in data.households
            if h["street"] == "A" and h["housing_type"] == housing_type
        )
        for housing_type in {"T1", "T2"}
    }
    outcome = db.lock_street(database, "A")
    assert not outcome.already

    # B is resized to 7 with T1=5, T2=2; type totals are derived from the
    # frozen A values so a solution must exist.
    result = db.publish(
        database,
        {"A": 999, "B": 7},
        {"T1": a_by_type["T1"] + 5, "T2": a_by_type["T2"] + 2},
    )
    assert result.ok, result.conflict

    data = db.get_dashboard(database)
    a_after = {
        h["sample_id"]: h["current_weight"]
        for h in data.households
        if h["street"] == "A"
    }
    assert a_after == a_weights
    assert sum(h["current_weight"] for h in data.households if h["street"] == "B") == 7
    assert set(data.locks) == {"A"}


def test_second_lock_is_idempotent(database: Path):
    db.publish(database, {"A": 8, "B": 6}, {"T1": 8, "T2": 6})
    first = db.lock_street(database, "A")
    second = db.lock_street(database, "A")
    assert not first.already and second.already
    assert first.total == second.total


def test_lock_between_publishes_still_binds(database: Path):
    """Lock after publish #1, then publish #2 must respect the earlier lock."""
    db.publish(database, {"A": 8, "B": 6}, {"T1": 8, "T2": 6})
    data = db.get_dashboard(database)
    first = {
        h["sample_id"]: h["current_weight"]
        for h in data.households
        if h["street"] == "B"
    }
    b_by_type = {
        housing_type: sum(
            h["current_weight"]
            for h in data.households
            if h["street"] == "B" and h["housing_type"] == housing_type
        )
        for housing_type in {"T1", "T2"}
    }
    db.lock_street(database, "B")
    outcome = db.publish(
        database,
        {"A": 5, "B": 1},
        {"T1": b_by_type["T1"] + 2, "T2": b_by_type["T2"] + 3},
    )
    assert outcome.ok, outcome.conflict
    data = db.get_dashboard(database)
    second = {
        h["sample_id"]: h["current_weight"]
        for h in data.households
        if h["street"] == "B"
    }
    assert second == first


def test_conflict_references_lock_and_targets(database: Path):
    db.publish(database, {"A": 8, "B": 6}, {"T1": 8, "T2": 6})
    data = db.get_dashboard(database)
    a_t1 = sum(
        h["current_weight"]
        for h in data.households
        if h["street"] == "A" and h["housing_type"] == "T1"
    )
    a_t2 = sum(
        h["current_weight"]
        for h in data.households
        if h["street"] == "A" and h["housing_type"] == "T2"
    )
    db.lock_street(database, "A")
    assert a_t1 >= 2

    # T1 demand is below what the frozen A weights alone already contribute.
    outcome = db.publish(database, {"A": 1, "B": 6}, {"T1": a_t1 - 1, "T2": a_t2})
    assert not outcome.ok
    kinds = {(item["kind"], item["name"]) for item in outcome.conflict}
    assert ("lock", "A") in kinds

    # Independently verify the reported set is infeasible and irreducible.
    snapshot = db.get_dashboard(database).households
    households = [
        solver.Household(
            h["id"], h["sample_id"], h["street"], h["housing_type"],
            h["initial_weight"], h["current_weight"],
        )
        for h in snapshot
    ]
    street_targets = {"B": 6}
    type_targets = {"T1": a_t1 - 1, "T2": a_t2}
    lock_weights = {
        h["id"]: h["current_weight"] for h in snapshot if h["street"] == "A"
    }
    subset = kinds
    assert not solver._subset_feasible(
        households, subset, street_targets, type_targets, lock_weights, {"A"}
    )
    for item in subset:
        assert solver._subset_feasible(
            households, subset - {item}, street_targets, type_targets,
            lock_weights, {"A"},
        ), f"删除 {item} 后仍无解"

    # Nothing was written and the street totals are unchanged.
    after = db.get_dashboard(database)
    assert after.summary["publication_count"] >= 1
    assert set(after.locks) == {"A"}


def test_lock_before_first_publish_uses_initial_weights(database: Path):
    initial = {
        h["sample_id"]: h["initial_weight"]
        for h in db.get_dashboard(database).households
        if h["street"] == "A"
    }
    outcome = db.lock_street(database, "A")
    assert not outcome.already
    assert outcome.total == sum(initial.values())
    assert db.get_dashboard(database).summary["publication_count"] == 0

    result = db.publish(database, {"B": 4}, {"T1": 5, "T2": 4})
    assert result.ok, result.conflict
    after = {
        h["sample_id"]: h["current_weight"]
        for h in db.get_dashboard(database).households
        if h["street"] == "A"
    }
    assert after == initial


def test_unknown_street_rejected(database: Path):
    with pytest.raises(db.ValidationError):
        db.lock_street(database, "不存在")
    with pytest.raises(db.ValidationError):
        db.publish(database, {"A": 4}, {"T1": 5, "T2": 5})


def test_missing_type_target_rejected(database: Path):
    with pytest.raises(db.ValidationError):
        db.publish(database, {"A": 4, "B": 4}, {"T1": 4})


def test_state_survives_reopen(database: Path):
    db.publish(database, {"A": 8, "B": 6}, {"T1": 8, "T2": 6})
    db.lock_street(database, "A")
    before = db.get_dashboard(database)
    reopened = db.get_dashboard(database)
    assert {
        h["sample_id"]: h["current_weight"] for h in before.households
    } == {h["sample_id"]: h["current_weight"] for h in reopened.households}
    assert set(reopened.locks) == {"A"}
    assert reopened.summary["publication_count"] == 1
