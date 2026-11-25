"""
app/db.py  -  database access for the Fitbite RAG.

Two ways for the RAG to use the DB:

1. retrieve_user_context(user_id) / build_context_text(user_id)
   Fixed, parameterised queries. Deterministic and safe. Use this to inject the
   user's data into the prompt on every chat turn.

2. run_safe_query(sql, user_id)
   Executes SQL written by the LLM (text-to-SQL) after strict validation.
   Use it as a tool the LLM can call for questions the fixed context doesn't cover.
"""

import re
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import settings


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_engine() -> Engine:
    if not settings.DATABASE_URL:
        raise ValueError("DATABASE_URL is missing in .env")

    return create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        # Every session is read-only and can't run longer than 5s.
        # (Works with psycopg2 / psycopg3 on PostgreSQL.)
        connect_args={
            "options": "-c default_transaction_read_only=on -c statement_timeout=5000"
        },
    )


def test_connection() -> bool:
    with get_engine().connect() as connection:
        connection.execute(text("SELECT 1"))
    return True


def _clean(value: Any) -> Any:
    """Make DB values JSON/prompt friendly."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _rows(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    with get_engine().connect() as conn:
        result = conn.execute(text(sql), params)
        return [{k: _clean(v) for k, v in row._mapping.items()} for row in result]


# --------------------------------------------------------------------------- #
# 1) Fixed, per-user context retrieval  (recommended default)
# --------------------------------------------------------------------------- #
def retrieve_user_context(user_id: int, days: int = 7) -> dict[str, Any]:
    p = {"uid": user_id, "days": days}

    profile = _rows(
        """
        SELECT name, age, height, gender, activity_level
        FROM profiles WHERE user_id = :uid
        """,
        p,
    )

    goal = _rows(
        """
        SELECT goal_type, target_weight, daily_calories, created_at
        FROM goals WHERE user_id = :uid
        ORDER BY created_at DESC NULLS LAST LIMIT 1
        """,
        p,
    )

    weights = _rows(
        """
        SELECT weight, recorded_at
        FROM weight_history WHERE user_id = :uid
        ORDER BY recorded_at DESC LIMIT 10
        """,
        p,
    )

    inbody = _rows(
        """
        SELECT measured_on, body_fat_pct, body_water_pct, protein_pct, bone_mass_kg
        FROM inbody_entries WHERE user_id = :uid
        ORDER BY measured_on DESC LIMIT 1
        """,
        p,
    )

    measurements = _rows(
        """
        SELECT measured_on, waist_cm, hips_cm, chest_cm, arm_cm, thigh_cm, calf_cm
        FROM body_measurements WHERE user_id = :uid
        ORDER BY measured_on DESC LIMIT 1
        """,
        p,
    )

    sleep = _rows(
        """
        SELECT sleep_date, hours
        FROM sleep_logs
        WHERE user_id = :uid AND sleep_date >= CURRENT_DATE - CAST(:days AS integer)
        ORDER BY sleep_date DESC
        """,
        p,
    )

    water = _rows(
        """
        SELECT log_date, SUM(amount_ml) AS total_ml
        FROM water_logs
        WHERE user_id = :uid AND log_date >= CURRENT_DATE - CAST(:days AS integer)
        GROUP BY log_date ORDER BY log_date DESC
        """,
        p,
    )

    # Calories/macros are only computed when the portion is in grams;
    # other units (piece, cup, ...) are returned as-is for the LLM to reason about.
    meals = _rows(
        """
        SELECT m.meal_date, m.meal_type, f.food_name, mi.portion, mi.portion_unit,
               CASE WHEN lower(coalesce(mi.portion_unit, 'g')) IN ('g', 'gram', 'grams')
                    THEN round(mi.portion / 100 * f.calories_per_100g) END AS kcal,
               CASE WHEN lower(coalesce(mi.portion_unit, 'g')) IN ('g', 'gram', 'grams')
                    THEN round(mi.portion / 100 * coalesce(f.protein_per_100g, 0), 1) END AS protein_g,
               CASE WHEN lower(coalesce(mi.portion_unit, 'g')) IN ('g', 'gram', 'grams')
                    THEN round(mi.portion / 100 * coalesce(f.carbs_per_100g, 0), 1) END AS carbs_g,
               CASE WHEN lower(coalesce(mi.portion_unit, 'g')) IN ('g', 'gram', 'grams')
                    THEN round(mi.portion / 100 * coalesce(f.fats_per_100g, 0), 1) END AS fats_g
        FROM meals m
        JOIN meal_items mi ON mi.meal_id = m.meal_id
        JOIN foods f       ON f.food_id  = mi.food_id
        WHERE m.user_id = :uid AND m.meal_date >= CURRENT_DATE - CAST(:days AS integer)
        ORDER BY m.meal_date DESC, m.meal_id, mi.item_id
        """,
        p,
    )

    plan = _rows(
        """
        SELECT dp.plan_id, dp.daily_calories, dp.goal_type, dp.notes, dp.source,
               dpm.day_index, dpm.meal_type, dpm.title,
               f.food_name, dpi.grams
        FROM diet_plans dp
        LEFT JOIN diet_plan_meals dpm ON dpm.plan_id = dp.plan_id
        LEFT JOIN diet_plan_items dpi ON dpi.plan_meal_id = dpm.plan_meal_id
        LEFT JOIN foods f             ON f.food_id = dpi.food_id
        WHERE dp.user_id = :uid
          AND dp.plan_id = (
              SELECT plan_id FROM diet_plans
              WHERE user_id = :uid AND is_active
              ORDER BY created_at DESC LIMIT 1
          )
        ORDER BY dpm.day_index, dpm.plan_meal_id, dpi.plan_item_id
        """,
        p,
    )

    return {
        "profile": profile[0] if profile else None,
        "goal": goal[0] if goal else None,
        "recent_weights": weights,
        "latest_inbody": inbody[0] if inbody else None,
        "latest_measurements": measurements[0] if measurements else None,
        "sleep": sleep,
        "water": water,
        "meals": meals,
        "active_diet_plan": plan,
        "window_days": days,
    }


def build_context_text(user_id: int, days: int = 7) -> str:
    """Compact text block ready to paste into the LLM prompt."""
    ctx = retrieve_user_context(user_id, days)
    lines: list[str] = ["# User data (from the Fitbite database)"]

    def section(title: str, data: Any) -> None:
        if not data:
            lines.append(f"## {title}\n(no data)")
            return
        lines.append(f"## {title}")
        if isinstance(data, dict):
            lines.append(", ".join(f"{k}={v}" for k, v in data.items() if v is not None))
        else:
            for row in data:
                lines.append("- " + ", ".join(f"{k}={v}" for k, v in row.items() if v is not None))

    section("Profile", ctx["profile"])
    section("Goal", ctx["goal"])
    section("Recent weights (newest first)", ctx["recent_weights"])
    section("Latest InBody", ctx["latest_inbody"])
    section("Latest body measurements", ctx["latest_measurements"])
    section(f"Sleep, last {days} days", ctx["sleep"])
    section(f"Water, last {days} days", ctx["water"])
    section(f"Meals, last {days} days", ctx["meals"])
    section("Active diet plan", ctx["active_diet_plan"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2) Guarded text-to-SQL
# --------------------------------------------------------------------------- #
# Tables the LLM may read. `password_hash`, `email`, `avatar_path` and
# `file_path` are deliberately not described to it and are blocked below.
USER_SCOPED_TABLES = {
    "profiles", "goals", "weight_history", "meals", "sleep_logs", "water_logs",
    "inbody_entries", "inbody_reports", "body_measurements", "diet_plans",
}
CHILD_TABLES = {"meal_items", "diet_plan_meals", "diet_plan_items"}  # no user_id column
GLOBAL_TABLES = {"foods"}
ALLOWED_TABLES = USER_SCOPED_TABLES | CHILD_TABLES | GLOBAL_TABLES

BLOCKED_COLUMNS = {"password_hash", "email", "avatar_path", "file_path"}
FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|call|do|"
    r"execute|set|reset|vacuum|analyze|listen|notify|lock|pg_sleep|into)\b",
    re.IGNORECASE,
)

SCHEMA_FOR_LLM = """
PostgreSQL schema (all tables in `public`). The current user is always :user_id
(bind parameter - write it literally as :user_id, never put a number there).

profiles(user_id PK, name, age, height, gender, activity_level, onboarding_done, created_at)
goals(goal_id PK, user_id FK, target_weight, daily_calories, goal_type, created_at)
weight_history(record_id PK, user_id FK, weight, recorded_at)
body_measurements(measure_id PK, user_id FK, measured_on, waist_cm, hips_cm, chest_cm, arm_cm, thigh_cm, calf_cm, created_at)
inbody_entries(entry_id PK, user_id FK, measured_on, body_fat_pct, body_water_pct, protein_pct, bone_mass_kg, created_at)
sleep_logs(log_id PK, user_id FK, sleep_date, hours)               -- unique (user_id, sleep_date)
water_logs(log_id PK, user_id FK, log_date, amount_ml, created_at)
meals(meal_id PK, user_id FK, meal_date, meal_type, notes)
meal_items(item_id PK, meal_id FK->meals, food_id FK->foods, portion, portion_unit)
foods(food_id PK, food_name, calories_per_100g, protein_per_100g, carbs_per_100g, fats_per_100g, name_ar)  -- shared, not per user
diet_plans(plan_id PK, user_id FK, daily_calories, goal_type, notes, source, is_active, created_at)
diet_plan_meals(plan_meal_id PK, plan_id FK->diet_plans, day_index, meal_type, title)
diet_plan_items(plan_item_id PK, plan_meal_id FK->diet_plan_meals, food_id FK->foods, grams)

Rules:
- One SELECT statement only. No SELECT *, list the columns you need.
- Every query touching a user table must filter with `user_id = :user_id`
  (for meal_items / diet_plan_* join up to meals / diet_plans and filter there).
- Always add a sensible date filter and LIMIT.
""".strip()

MAX_ROWS = 100


class UnsafeQueryError(ValueError):
    pass


def validate_sql(sql: str) -> str:
    q = sql.strip().rstrip(";").strip()

    if not q:
        raise UnsafeQueryError("Empty query.")
    if ";" in q:
        raise UnsafeQueryError("Only a single statement is allowed.")
    if "--" in q or "/*" in q:
        raise UnsafeQueryError("Comments are not allowed.")
    if not re.match(r"^(select|with)\b", q, re.IGNORECASE):
        raise UnsafeQueryError("Only SELECT queries are allowed.")
    if FORBIDDEN_KEYWORDS.search(q):
        raise UnsafeQueryError("Query contains a forbidden keyword.")
    if re.search(r"\b(pg_\w+|information_schema|current_setting|set_config)\b", q, re.IGNORECASE):
        raise UnsafeQueryError("System catalogs and functions are not allowed.")

    # No SELECT * / t.*  (it would leak password_hash etc.). count(*) is fine.
    if "*" in re.sub(r"count\s*\(\s*\*\s*\)", "", q, flags=re.IGNORECASE):
        raise UnsafeQueryError("Don't use *; list the columns you need.")

    for col in BLOCKED_COLUMNS:
        if re.search(rf"\b{col}\b", q, re.IGNORECASE):
            raise UnsafeQueryError(f"Column '{col}' is not accessible.")

    # CTE names are allowed as table references.
    cte_names = set(re.findall(r"\b(\w+)\s+as\s*\(", q, flags=re.IGNORECASE))
    tables = {
        t.split(".")[-1].lower()
        for t in re.findall(r"\b(?:from|join)\s+([\w.\"]+)", q, flags=re.IGNORECASE)
    }
    tables = {t.strip('"') for t in tables} - {c.lower() for c in cte_names}

    bad = tables - ALLOWED_TABLES
    if bad:
        raise UnsafeQueryError(f"Table(s) not allowed: {', '.join(sorted(bad))}")

    if (tables & (USER_SCOPED_TABLES | CHILD_TABLES)) and ":user_id" not in q:
        raise UnsafeQueryError("Query must filter by the current user with :user_id.")
    if ":user_id" in q and not re.search(r"user_id\s*=\s*:user_id", q, re.IGNORECASE):
        raise UnsafeQueryError("Use the form `user_id = :user_id`.")

    return q


def run_safe_query(sql: str, user_id: int) -> list[dict[str, Any]]:
    """Validate and run LLM-generated SQL, scoped to `user_id`, read-only, row-capped."""
    q = validate_sql(sql)
    wrapped = f"SELECT * FROM ({q}) AS _q LIMIT {MAX_ROWS}"
    return _rows(wrapped, {"user_id": user_id})
