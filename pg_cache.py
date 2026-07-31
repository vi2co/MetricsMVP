import os
from typing import Any

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()


class PgCache:
    def __init__(self):
        self.host = os.getenv("PG_HOST", "127.0.0.1")
        self.user = os.getenv("PG_USER", "test_user")
        self.password = os.getenv("PG_PASSWORD", "test_password")
        self.dbname = os.getenv("PG_DB", "postgres")
        self.port = int(os.getenv("PG_PORT", "5432"))
        self._conn = None

    def _connect(self):
        if self._conn is None or self._conn.closed:
            try:
                self._conn = psycopg2.connect(
                    host=self.host,
                    user=self.user,
                    password=self.password,
                    dbname=self.dbname,
                    port=self.port,
                )
                self._conn.autocommit = True
            except Exception as e:
                print(f"PG connection failed: {e}")
                self._conn = None

    def init_db(self):
        self._connect()
        if self._conn is None:
            print("PG unavailable — falling back to in-memory cache only.")
            return

        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS vehicle_cache (
                        id SERIAL PRIMARY KEY,
                        year VARCHAR(4) NOT NULL,
                        make VARCHAR(100) NOT NULL,
                        model VARCHAR(100) NOT NULL,
                        trim VARCHAR(100),
                        length NUMERIC,
                        width NUMERIC,
                        height NUMERIC,
                        wheelbase NUMERIC,
                        ground_clearance NUMERIC,
                        curb_weight NUMERIC,
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(year, make, model, trim)
                    )
                """)
            print("PG cache table ready.")
        except Exception as e:
            print(f"PG init failed: {e}")

    def get_dimensions(
        self,
        year: str | int,
        make: str,
        model: str,
        trim: str | None = None,
    ) -> dict[str, Any] | None:
        self._connect()
        if self._conn is None:
            return None

        try:
            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT year, make, model, trim,
                           length, width, height, wheelbase,
                           ground_clearance, curb_weight
                    FROM vehicle_cache
                    WHERE year = %s
                      AND LOWER(make) = LOWER(%s)
                      AND LOWER(model) = LOWER(%s)
                      AND (
                          (trim IS NULL AND %s IS NULL)
                          OR LOWER(trim) = LOWER(%s)
                      )
                    """,
                    (str(year), make, model, trim, trim),
                )
                row = cur.fetchone()
                if row:
                    return dict(row)
            return None
        except Exception as e:
            print(f"PG read failed: {e}")
            return None

    def set_dimensions(
        self,
        year: str | int,
        make: str,
        model: str,
        trim: str | None,
        dimensions: dict[str, Any] | None,
    ) -> None:
        if dimensions is None:
            return

        self._connect()
        if self._conn is None:
            return

        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO vehicle_cache
                        (year, make, model, trim,
                         length, width, height, wheelbase,
                         ground_clearance, curb_weight)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (year, make, model, trim)
                    DO UPDATE SET
                        length = EXCLUDED.length,
                        width = EXCLUDED.width,
                        height = EXCLUDED.height,
                        wheelbase = EXCLUDED.wheelbase,
                        ground_clearance = EXCLUDED.ground_clearance,
                        curb_weight = EXCLUDED.curb_weight,
                        created_at = NOW()
                    """,
                    (
                        str(year),
                        make,
                        model,
                        trim,
                        dimensions.get("length"),
                        dimensions.get("width"),
                        dimensions.get("height"),
                        dimensions.get("wheelbase"),
                        dimensions.get("ground_clearance"),
                        dimensions.get("curb_weight"),
                    ),
                )
        except Exception as e:
            print(f"PG write failed: {e}")

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
