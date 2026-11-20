"""FastAPI web application for household energy-survey weight calibration."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Mapping

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from . import db

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _db_path() -> Path:
    return Path(os.environ.get("CALIBRATION_DB", str(BASE_DIR / "data" / "calibration.sqlite3")))


def _sample_path() -> Path:
    return Path(
        os.environ.get("CALIBRATION_SAMPLE", str(BASE_DIR / "data" / "sample.csv"))
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.initialize(_db_path(), _sample_path())
    yield


app = FastAPI(title="住户能源调查权重校准", lifespan=lifespan)
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def _render(
    request: Request,
    status_code: int = 200,
    errors: List[str] | None = None,
    form: Mapping[str, str] | None = None,
    conflict: List[dict] | None = None,
    conflict_message: str | None = None,
    scroll_to: str | None = None,
) -> HTMLResponse:
    data = db.get_dashboard(_db_path())
    notices: List[str] = []
    notice = request.query_params.get("notice")
    if notice:
        notices.append(notice)
    context = {
        "request": request,
        "data": data,
        "errors": errors or [],
        "notices": notices,
        "form": dict(form or {}),
        "conflict": conflict or [],
        "conflict_message": conflict_message,
        "scroll_to": scroll_to,
    }
    return templates.TemplateResponse(request, "index.html", context, status_code=status_code)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return _render(request)


@app.post("/lock")
def lock(
    request: Request,
    street: str = Form(default=""),
):
    try:
        outcome = db.lock_street(_db_path(), street.strip())
    except db.ValidationError as exc:
        return _render(
            request,
            status_code=400,
            errors=[str(exc)],
            form={"street": street},
            scroll_to="locks",
        )
    message = (
        f"街道「{outcome.street}」此前已锁定，沿用原锁定权重（{outcome.count} 户，合计 {outcome.total:,}）。"
        if outcome.already
        else f"已锁定街道「{outcome.street}」：{outcome.count} 户，合计 {outcome.total:,}。"
    )
    return RedirectResponse(url="/?notice=" + _quote(message) + "#locks", status_code=303)


def _quote(text: str) -> str:
    from urllib.parse import quote

    return quote(text, safe="")


def _parse_targets(
    form: Mapping[str, str], names: List[str], label: str
) -> Dict[str, int]:
    targets: Dict[str, int] = {}
    problems: List[str] = []
    for name in names:
        raw = (form.get(f"target__{name}") or "").strip()
        if not raw:
            problems.append(f"{label}「{name}」的目标户数未填写")
            continue
        if not raw.isdigit():
            problems.append(f"{label}「{name}」的目标户数必须是非负整数：{raw!r}")
            continue
        value = int(raw)
        if value > db.MAX_TARGET:
            problems.append(f"{label}「{name}」的目标户数过大（上限 {db.MAX_TARGET:,}）")
            continue
        targets[name] = value
    if problems:
        raise db.ValidationError("；".join(problems))
    return targets


def _form_snapshot(form: Mapping[str, str]) -> Dict[str, str]:
    return {key: value for key, value in form.items() if key.startswith("target__")}


@app.post("/publish")
async def publish(request: Request):
    form = await request.form()
    form_dict = _form_snapshot(form)
    data = db.get_dashboard(_db_path())
    # Locked streets have frozen weights and an implied total, so their form
    # value is not a user-editable condition and need not be validated.
    street_names = [street["name"] for street in data.streets if not street["locked"]]
    type_names = [group["name"] for group in data.types]
    try:
        street_targets = _parse_targets(form, street_names, "街道")
        type_targets = _parse_targets(form, type_names, "住房类型")
    except db.ValidationError as exc:
        return _render(
            request,
            status_code=400,
            errors=[str(exc)],
            form=form_dict,
            scroll_to="publish",
        )

    try:
        outcome = db.publish(_db_path(), street_targets, type_targets)
    except db.ValidationError as exc:
        return _render(
            request,
            status_code=400,
            errors=[str(exc)],
            form=form_dict,
            scroll_to="publish",
        )

    if not outcome.ok:
        return _render(
            request,
            status_code=409,
            conflict=outcome.conflict,
            conflict_message="当前目标与锁定条件无解，以下是一组极小冲突（去掉其中任意一项，其余条件即可同时满足）：",
            form=form_dict,
            scroll_to="publish",
        )

    message = (
        f"发布成功：{outcome.updated} 户已更新，现行权重合计 {outcome.total:,}"
        f"（发布记录 #{outcome.publication_id}）。"
    )
    return RedirectResponse(url="/?notice=" + _quote(message) + "#summary", status_code=303)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}
