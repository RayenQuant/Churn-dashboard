"""
pipeline.py — Data Engineering Pipeline
Ingests CSV → SQLite with proper schema, indexes, and analytical SQL views.
All analytics queries go through SQL (pandas.read_sql_query).
"""

import sqlite3
import os
import numpy as np
import pandas as pd
from faker import Faker

DB_PATH = "churn.db"
CSV_PATH = "customer_churn.csv"

# ---------------------------------------------------------------------------
# Synthetic data generation (runs only when CSV is missing)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(path: str = CSV_PATH, n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic 10,000-row synthetic churn dataset matching Kaggle schema."""
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    geography = rng.choice(["France", "Germany", "Spain"], size=n, p=[0.50, 0.25, 0.25])
    gender = rng.choice(["Male", "Female"], size=n, p=[0.545, 0.455])
    age = rng.integers(18, 93, size=n)
    tenure = rng.integers(0, 11, size=n)
    credit_score = np.clip(rng.normal(650, 97, size=n).astype(int), 300, 850)
    balance = np.where(rng.random(n) < 0.25, 0.0, np.abs(rng.normal(76_000, 62_000, size=n)).round(2))
    num_products = rng.choice([1, 2, 3, 4], size=n, p=[0.50, 0.33, 0.12, 0.05])
    has_cr_card = rng.choice([0, 1], size=n, p=[0.30, 0.70])
    is_active = rng.choice([0, 1], size=n, p=[0.485, 0.515])
    salary = np.abs(rng.normal(100_000, 57_000, size=n)).round(2)

    # Churn probability model (realistic drivers)
    logit = (
        -1.5
        + 0.04 * (age - 35)
        - 0.08 * tenure
        + 0.6 * (np.array(geography) == "Germany").astype(float)
        + 0.4 * (num_products >= 3).astype(float)
        - 0.5 * is_active
        + 0.3 * (np.array(gender) == "Female").astype(float)
        - 0.002 * (credit_score - 600)
        + rng.normal(0, 0.5, size=n)
    )
    prob = 1 / (1 + np.exp(-logit))
    exited = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "RowNumber": np.arange(1, n + 1),
        "CustomerId": [fake.unique.random_int(min=10000000, max=99999999) for _ in range(n)],
        "Surname": [fake.last_name() for _ in range(n)],
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary,
        "Exited": exited,
    })
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# SQLite ingestion
# ---------------------------------------------------------------------------

def ingest_csv_to_sqlite(csv_path: str = CSV_PATH, db_path: str = DB_PATH) -> None:
    """Load CSV into SQLite with typed schema and useful indexes."""
    if not os.path.exists(csv_path):
        generate_synthetic_dataset(csv_path)

    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS customers")
    cur.execute("""
        CREATE TABLE customers (
            RowNumber       INTEGER PRIMARY KEY,
            CustomerId      INTEGER NOT NULL,
            Surname         TEXT,
            CreditScore     INTEGER,
            Geography       TEXT,
            Gender          TEXT,
            Age             INTEGER,
            Tenure          INTEGER,
            Balance         REAL,
            NumOfProducts   INTEGER,
            HasCrCard       INTEGER,
            IsActiveMember  INTEGER,
            EstimatedSalary REAL,
            Exited          INTEGER
        )
    """)

    df.to_sql("customers", conn, if_exists="append", index=False)

    # Indexes for common query patterns
    cur.execute("CREATE INDEX IF NOT EXISTS idx_geography ON customers(Geography)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_exited ON customers(Exited)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_products ON customers(NumOfProducts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_active ON customers(IsActiveMember)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_gender ON customers(Gender)")

    # -----------------------------------------------------------------------
    # Analytical SQL views
    # -----------------------------------------------------------------------

    cur.execute("DROP VIEW IF EXISTS v_churn_by_tenure")
    cur.execute("""
        CREATE VIEW v_churn_by_tenure AS
        SELECT
            Tenure,
            COUNT(*)                          AS total_customers,
            SUM(Exited)                       AS churned,
            ROUND(AVG(Exited) * 100, 2)       AS churn_rate_pct
        FROM customers
        GROUP BY Tenure
        ORDER BY Tenure
    """)

    cur.execute("DROP VIEW IF EXISTS v_churn_by_geography")
    cur.execute("""
        CREATE VIEW v_churn_by_geography AS
        SELECT
            Geography,
            COUNT(*)                          AS total_customers,
            SUM(Exited)                       AS churned,
            ROUND(AVG(Exited) * 100, 2)       AS churn_rate_pct
        FROM customers
        GROUP BY Geography
        ORDER BY churn_rate_pct DESC
    """)

    cur.execute("DROP VIEW IF EXISTS v_churn_by_products")
    cur.execute("""
        CREATE VIEW v_churn_by_products AS
        SELECT
            NumOfProducts,
            COUNT(*)                          AS total_customers,
            SUM(Exited)                       AS churned,
            ROUND(AVG(Exited) * 100, 2)       AS churn_rate_pct
        FROM customers
        GROUP BY NumOfProducts
        ORDER BY NumOfProducts
    """)

    cur.execute("DROP VIEW IF EXISTS v_high_value_at_risk")
    cur.execute("""
        CREATE VIEW v_high_value_at_risk AS
        SELECT
            CustomerId, Surname, Geography, Gender, Age,
            Balance, CreditScore, NumOfProducts, IsActiveMember,
            EstimatedSalary
        FROM customers
        WHERE Exited = 1
          AND Balance > (SELECT AVG(Balance) FROM customers)
        ORDER BY Balance DESC
    """)

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# SQL query helpers (all analytics go through SQL)
# ---------------------------------------------------------------------------

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def sql_query(query: str, params: tuple = (), db_path: str = DB_PATH) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame."""
    conn = get_connection(db_path)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def build_where_clause(geography: list | None, gender: list | None,
                       num_products: list | None, is_active: list | None) -> str:
    """Build a dynamic WHERE clause from sidebar filters."""
    clauses = []
    if geography:
        vals = ",".join(f"'{g}'" for g in geography)
        clauses.append(f"Geography IN ({vals})")
    if gender:
        vals = ",".join(f"'{g}'" for g in gender)
        clauses.append(f"Gender IN ({vals})")
    if num_products:
        vals = ",".join(str(n) for n in num_products)
        clauses.append(f"NumOfProducts IN ({vals})")
    if is_active is not None and len(is_active) > 0:
        vals = ",".join(str(a) for a in is_active)
        clauses.append(f"IsActiveMember IN ({vals})")
    return " WHERE " + " AND ".join(clauses) if clauses else ""


# ---------------------------------------------------------------------------
# Filtered SQL queries for dashboard (Tab 1 & Tab 2)
# ---------------------------------------------------------------------------

def query_kpis(where: str = "") -> dict:
    """Return KPI values via SQL."""
    base = f"SELECT * FROM customers{where}"
    q = f"""
        SELECT
            COUNT(*)                                            AS total_customers,
            ROUND(AVG(Exited) * 100, 2)                        AS churn_rate,
            ROUND(AVG(CASE WHEN Exited=1 THEN Balance END), 2) AS avg_balance_churned,
            ROUND(SUM(CASE WHEN Exited=1 THEN EstimatedSalary * 0.15 ELSE 0 END), 2) AS revenue_at_risk
        FROM ({base})
    """
    return sql_query(q).iloc[0].to_dict()


def query_churn_donut(where: str = "") -> pd.DataFrame:
    return sql_query(f"""
        SELECT
            CASE WHEN Exited=1 THEN 'Churned' ELSE 'Retained' END AS status,
            COUNT(*) AS count
        FROM customers{where}
        GROUP BY Exited
    """)


def query_churn_by_geography(where: str = "") -> pd.DataFrame:
    return sql_query(f"""
        SELECT Geography,
               ROUND(AVG(Exited)*100, 2) AS churn_rate_pct,
               COUNT(*) AS total
        FROM customers{where}
        GROUP BY Geography ORDER BY churn_rate_pct DESC
    """)


def query_churn_by_products(where: str = "") -> pd.DataFrame:
    return sql_query(f"""
        SELECT NumOfProducts,
               ROUND(AVG(Exited)*100, 2) AS churn_rate_pct,
               COUNT(*) AS total
        FROM customers{where}
        GROUP BY NumOfProducts ORDER BY NumOfProducts
    """)


def query_cumulative_churn_by_tenure(where: str = "") -> pd.DataFrame:
    return sql_query(f"""
        SELECT Tenure,
               SUM(Exited) AS churned,
               COUNT(*) AS total
        FROM customers{where}
        GROUP BY Tenure ORDER BY Tenure
    """)


def query_all_customers(where: str = "") -> pd.DataFrame:
    return sql_query(f"SELECT * FROM customers{where}")


# ---------------------------------------------------------------------------
# Ensure DB exists on import
# ---------------------------------------------------------------------------

def ensure_db():
    if not os.path.exists(DB_PATH):
        ingest_csv_to_sqlite()
