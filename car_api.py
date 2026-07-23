import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


# CarAPI free-tier data covers these model years.
SUPPORTED_YEAR_MIN = 2015
SUPPORTED_YEAR_MAX = 2020

# Classifier model names sometimes differ from CarAPI model names.
MODEL_ALIASES = {
    "silverado": "Silverado 1500",
    "sierra": "Sierra 1500",
    "ram": "1500",
    # Lexus embeds displacement in the CarAPI model name (no space).
    "lx": "LX570",
    "rx": "RX350",
    "gx": "GX460",
}

# Body-style words the classifier appends that CarAPI models do not use.
BODY_STYLE_SUFFIXES = {
    "sedan",
    "coupe",
    "suv",
    "hatchback",
    "wagon",
    "van",
    "minivan",
    "convertible",
    "pickup",
    "crew",
    "cab",
    "extended",
    "regular",
}


def normalize_model_name(model: str) -> str:
    words = model.strip().split()

    while words and words[-1].lower() in BODY_STYLE_SUFFIXES:
        words.pop()

    cleaned = " ".join(words) or model.strip()
    return MODEL_ALIASES.get(cleaned.lower(), cleaned)


def clamp_year(year: str | int) -> str:
    """Clamp a predicted year into the supported CarAPI data range."""
    try:
        value = int(str(year).strip())
    except ValueError:
        return str(year)

    return str(min(max(value, SUPPORTED_YEAR_MIN), SUPPORTED_YEAR_MAX))


class CarAPI:
    BASE_URL = "https://carapi.app/api"

    def __init__(self) -> None:
        self.api_token = os.getenv("CARAPI_TOKEN", "").strip()
        self.api_secret = os.getenv("CARAPI_SECRET", "").strip()
        self.jwt: str | None = None
        self.cache: dict[tuple[str, str, str, str], dict[str, Any] | None] = {}

        if not self.api_token:
            raise RuntimeError("Missing CARAPI_TOKEN in .env")

        if not self.api_secret:
            raise RuntimeError("Missing CARAPI_SECRET in .env")

    def authenticate(self) -> str:
        print("Connecting to CarAPI...")

        response = requests.post(
            f"{self.BASE_URL}/auth/login",
            json={
                "api_token": self.api_token,
                "api_secret": self.api_secret,
            },
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=20,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"CarAPI authentication failed "
                f"({response.status_code}): {response.text}"
            )

        self.jwt = response.text.strip().strip('"')

        if not self.jwt:
            raise RuntimeError("CarAPI returned an empty JWT.")

        print("CarAPI authenticated successfully.")
        return self.jwt

    def headers(self) -> dict[str, str]:
        if not self.jwt:
            self.authenticate()

        return {
            "Authorization": f"Bearer {self.jwt}",
            "Accept": "application/json",
        }

    def get_dimensions(
        self,
        year: str | int,
        make: str,
        model: str,
        trim: str | None = None,
    ) -> dict[str, Any] | None:
        cache_key = (
            str(year).strip(),
            make.lower().strip(),
            model.lower().strip(),
            (trim or "").lower().strip(),
        )

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Use a representative year within the supported data range
        # and normalize the model name to CarAPI conventions.
        lookup_year = clamp_year(year)
        lookup_model = normalize_model_name(model)

        params = {
            "year": lookup_year,
            "make": make,
            "model": lookup_model,
            "limit": 10,
            "verbose": "yes",
        }

        if trim:
            params["trim"] = trim

        print(
            f"Searching CarAPI for {lookup_year} {make} {lookup_model}"
            f"{f' {trim}' if trim else ''}"
            f" (predicted: {year} {model})..."
        )

        records = self._request_body_records(params)

        # Retry without trim if the representative trim is unavailable.
        if not records and trim:
            print(
                f"No exact trim match for {lookup_year} {make} "
                f"{lookup_model} {trim}. Retrying with model only..."
            )

            params.pop("trim", None)
            records = self._request_body_records(params)

        if not records:
            print(f"No body record found for {lookup_year} {make} {lookup_model}.")
            self.cache[cache_key] = None
            return None

        body = records[0]

        dimensions = {
            "year": body.get("year", year),
            "make": body.get("make", make),
            "model": body.get("model", model),
            "trim": body.get("trim") or trim or "Representative",
            "length": body.get("length"),
            "width": body.get("width"),
            "height": body.get("height"),
            "wheelbase": body.get("wheel_base"),
            "ground_clearance": body.get("ground_clearance"),
            "curb_weight": body.get("curb_weight"),
        }

        self.cache[cache_key] = dimensions
        return dimensions

    def _request_body_records(
        self,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            response = requests.get(
                f"{self.BASE_URL}/bodies/v2",
                headers=self.headers(),
                params=params,
                timeout=20,
            )

            if response.status_code != 200:
                print(
                    f"CarAPI lookup failed ({response.status_code}): "
                    f"{response.text}"
                )
                return []

            return self._extract_records(response.json())

        except requests.Timeout:
            print("CarAPI dimension lookup timed out.")
        except requests.RequestException as error:
            print(f"CarAPI request failed: {error}")
        except ValueError:
            print("CarAPI returned invalid JSON.")

        return []

    @staticmethod
    def _extract_records(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return data

        if not isinstance(data, dict):
            return []

        for key in ("data", "collection", "bodies", "items", "results"):
            records = data.get(key)

            if isinstance(records, list):
                return records

            if isinstance(records, dict):
                for nested_key in ("items", "data", "results"):
                    nested_records = records.get(nested_key)

                    if isinstance(nested_records, list):
                        return nested_records

        if any(
            key in data
            for key in (
                "length",
                "width",
                "height",
                "wheel_base",
                "curb_weight",
            )
        ):
            return [data]

        return []

    def test_connection(self) -> bool:
        self.authenticate()
        return True


if __name__ == "__main__":
    api = CarAPI()
    api.authenticate()

    dimensions = api.get_dimensions(
        year=2020,
        make="Toyota",
        model="Camry",
        trim="LE",
    )

    print(dimensions)