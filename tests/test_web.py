"""End-to-end tests of the FastAPI routes using an isolated database."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db, main


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sample = tmp_path / "sample.csv"
    with sample.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["样本号", "街道", "住房类型", "初始权重"])
        writer.writerows(
            [
                ("H01", "A", "T1", 2),
                ("H02", "A", "T2", 3),
                ("H03", "B", "T1", 4),
                ("H04", "B", "T2", 1),
            ]
        )
    monkeypatch.setenv("CALIBRATION_DB", str(tmp_path / "web.sqlite3"))
    monkeypatch.setenv("CALIBRATION_SAMPLE", str(sample))
    with TestClient(main.app) as test_client:
        yield test_client


def targets_from(html: str) -> dict:
    import re

    return {
        name: value
        for name, value in re.findall(
            r'name="target__([^"]+)"[^>]*value="(\d+)"', html, re.S
        )
    }


def editable_from(html: str) -> list:
    import re

    return re.findall(r'name="editable__([^"]+)"', html)


def publish_form(values: dict, editable: list) -> dict:
    """Build a submission the way a browser would: targets plus the hidden
    markers recording which streets were editable when the form rendered."""
    data = {f"target__{name}": value for name, value in values.items()}
    data.update({f"editable__{name}": "1" for name in editable})
    return data


def test_index_shows_initial_state(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "台账概览" in response.text
    assert "A" in response.text
    values = targets_from(response.text)
    assert values == {"A": "5", "B": "5", "T1": "6", "T2": "4"}


def test_publish_round_trip(client: TestClient):
    page = client.get("/").text
    values = targets_from(page)
    response = client.post(
        "/publish",
        data={f"target__{name}": value for name, value in values.items()},
    )
    assert response.status_code == 200
    assert "发布成功" in response.text
    assert "成功发布次数" in response.text
    assert "现行权重合计" in response.text


def test_publish_rejects_non_integer(client: TestClient):
    values = targets_from(client.get("/").text)
    values["A"] = "abc"
    response = client.post(
        "/publish", data={f"target__{name}": value for name, value in values.items()}
    )
    assert response.status_code == 400
    assert "非负整数" in response.text


def test_publish_conflict_returns_409_and_keeps_weights(client: TestClient):
    before = client.get("/").text
    values = targets_from(before)
    values["T1"] = "99"
    response = client.post(
        "/publish", data={f"target__{name}": value for name, value in values.items()}
    )
    assert response.status_code == 409
    assert "无解" in response.text
    assert "<li>" in response.text
    after = client.get("/").text
    assert targets_from(after) == targets_from(before)


def test_lock_then_publish_keeps_locked_weights(client: TestClient):
    values = targets_from(client.get("/").text)
    client.post(
        "/publish", data={f"target__{name}": value for name, value in values.items()}
    )
    page = client.get("/").text
    assert "锁定该街道现行权重" in page

    response = client.post("/lock", data={"street": "A"})
    assert response.status_code == 200
    assert "已锁定" in response.text

    # The locked street's form value is readonly and its target is ignored.
    page = client.get("/").text
    assert "已锁定" in page
    data = db.get_dashboard(main._db_path())
    locked = [h for h in data.households if h["street"] == "A"]
    a_t1 = sum(h["current_weight"] for h in locked if h["housing_type"] == "T1")
    a_t2 = sum(h["current_weight"] for h in locked if h["housing_type"] == "T2")

    values = targets_from(page)
    values["A"] = "1"  # ignored: locked streets keep their implied total
    values["B"] = "6"
    values["T1"] = str(a_t1 + 5)
    values["T2"] = str(a_t2 + 1)
    response = client.post(
        "/publish", data={f"target__{name}": value for name, value in values.items()}
    )
    assert response.status_code == 200
    assert "发布成功" in response.text

    after = db.get_dashboard(main._db_path())
    assert {
        h["sample_id"]: h["current_weight"]
        for h in after.households
        if h["street"] == "A"
    } == {h["sample_id"]: h["current_weight"] for h in locked}


def test_interleaved_lock_rejects_stale_target(client: TestClient):
    """Tab A fills targets; tab B locks a street; tab A submits the stale form."""
    # Tab A: render the form and fill in targets (all streets editable).
    page = client.get("/").text
    editable = editable_from(page)
    assert set(editable) == {"A", "B"}
    values = targets_from(page)
    values.update({"A": "8", "B": "4", "T1": "7", "T2": "5"})  # feasible together

    # Tab B: lock street A (still at initial weights, locked total 5).
    assert client.post("/lock", data={"street": "A"}).status_code == 200

    # Tab A: submit the originally filled form.
    response = client.post("/publish", data=publish_form(values, editable))
    assert response.status_code == 409
    assert "发布被拒绝" in response.text
    assert "原填目标户数" in response.text
    assert "锁定合计" in response.text
    assert "A" in response.text
    # The page lists the originally filled target and the locked total.
    assert ">8</td>" in response.text
    assert ">5</td>" in response.text

    # Current weights and publication records are unchanged.
    data = db.get_dashboard(main._db_path())
    assert all(
        h["current_weight"] == h["initial_weight"] for h in data.households
    )
    assert data.summary["publication_count"] == 0

    # Still-editable targets are kept for revision; the newly locked street
    # shows its locked total instead of the stale filled value.
    shown = targets_from(response.text)
    assert shown["B"] == "4"
    assert shown["T1"] == "7"
    assert shown["T2"] == "5"
    assert shown["A"] == "5"

    # Revise the remaining editable targets against the lock and republish.
    page = client.get("/").text
    editable = editable_from(page)
    assert editable == ["B"]
    values = targets_from(page)
    values.update({"B": "8", "T1": "8", "T2": "5"})
    response = client.post("/publish", data=publish_form(values, editable))
    assert response.status_code == 200
    assert "发布成功" in response.text
    data = db.get_dashboard(main._db_path())
    assert {
        h["sample_id"]: h["current_weight"]
        for h in data.households
        if h["street"] == "A"
    } == {"H01": 2, "H02": 3}
    assert sum(
        h["current_weight"] for h in data.households if h["street"] == "B"
    ) == 8
    assert data.summary["publication_count"] == 1


def test_interleaved_lock_matching_target_publishes(client: TestClient):
    """A new lock consistent with the filled target publishes as usual."""
    page = client.get("/").text
    editable = editable_from(page)
    values = targets_from(page)  # defaults: A=5, B=5, T1=6, T2=4

    # Tab B locks A at its current (initial) weights: locked total 5, which
    # matches the target filled in tab A.
    assert client.post("/lock", data={"street": "A"}).status_code == 200

    response = client.post("/publish", data=publish_form(values, editable))
    assert response.status_code == 200
    assert "发布成功" in response.text
    data = db.get_dashboard(main._db_path())
    assert {
        h["sample_id"]: h["current_weight"]
        for h in data.households
        if h["street"] == "A"
    } == {"H01": 2, "H02": 3}
    assert data.summary["publication_count"] == 1


def test_lock_unknown_street_shows_error(client: TestClient):
    response = client.post("/lock", data={"street": "缺失街道"})
    assert response.status_code == 400
    assert "街道不存在" in response.text


def test_state_readable_after_app_restart(client: TestClient, monkeypatch):
    values = targets_from(client.get("/").text)
    client.post(
        "/publish", data={f"target__{name}": value for name, value in values.items()}
    )
    client.post("/lock", data={"street": "B"})
    db_path = main._db_path()
    sample_path = main._sample_path()
    # A brand new app instance over the same SQLite file.
    with TestClient(main.app) as restarted:
        page = restarted.get("/").text
        assert "已锁定" in page
        assert "成功发布次数" in page
        assert "B" in page
    assert db_path.exists() and sample_path.exists()


def test_all_streets_locked_renders_and_type_only_publish(client: TestClient):
    for street in ("A", "B"):
        assert client.post("/lock", data={"street": street}).status_code == 200
    page = client.get("/").text
    assert "所有街道均已锁定" in page
    # With every street frozen, only type targets remain adjustable; use the
    # frozen totals to build a satisfiable combination.
    data = db.get_dashboard(main._db_path())
    by_type = {
        name: sum(h["current_weight"] for h in data.households if h["housing_type"] == name)
        for name in ("T1", "T2")
    }
    response = client.post(
        "/publish",
        data={"target__T1": str(by_type["T1"]), "target__T2": str(by_type["T2"])},
    )
    assert response.status_code == 200
    assert "发布成功" in response.text


def test_healthz(client: TestClient):
    assert client.get("/healthz").json() == {"status": "ok"}
