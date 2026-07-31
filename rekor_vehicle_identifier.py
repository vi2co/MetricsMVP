import os

import cv2
import requests
from dotenv import load_dotenv


load_dotenv()


class RekorVehicleIdentifier:
    API_URL = "https://api.openalpr.com/v3/recognize"

    def __init__(self):
        self.secret_key = os.getenv("REKOR_SECRET_KEY", "").strip()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MetricsMVP/1.0",
        })

        if not self.secret_key:
            raise RuntimeError(
                "Missing REKOR_SECRET_KEY in .env — "
                "get yours at https://cloud.openalpr.com/account/register"
            )

    def classify(self, crop, track_id=None):
        if crop is None or crop.size == 0:
            return "Unknown", 0.0

        if min(crop.shape[:2]) < 60:
            return "Unknown", 0.0

        success, buffer = cv2.imencode(".jpg", crop)

        if not success:
            print("Failed to encode vehicle crop.")
            return "Unknown", 0.0

        data = self._post_image(buffer.tobytes())

        if not data:
            return "Unknown", 0.0

        return self._parse_vehicle(data)

    def _post_image(self, image_bytes: bytes) -> dict | None:
        params = {
            "recognize_vehicle": 1,
            "country": "us",
            "secret_key": self.secret_key,
        }

        try:
            response = self.session.post(
                self.API_URL,
                params=params,
                files={"image": ("crop.jpg", image_bytes, "image/jpeg")},
                timeout=30,
            )

            if response.status_code == 403:
                print(
                    "Rekor API returned 403 Forbidden. This usually means the "
                    "secret key is invalid, expired, or the account has no active "
                    "subscription. Check REKOR_SECRET_KEY in .env"
                )
                return None

            if response.status_code == 402:
                print(
                    "Rekor API returned 402 Payment Required. "
                    "Your API credit balance may be exhausted."
                )
                return None

            if response.status_code != 200:
                print(
                    f"Rekor API error ({response.status_code}): {response.text[:500]}"
                )
                return None

            return response.json()

        except requests.Timeout:
            print("Rekor API request timed out.")
        except requests.RequestException as error:
            print(f"Rekor API request failed: {error}")
        except ValueError:
            print("Rekor API returned invalid JSON.")

        return None

    def _parse_vehicle(self, data: dict) -> tuple[str, float]:
        results = data.get("results") if isinstance(data.get("results"), list) else []
        vehicles = data.get("vehicles") if isinstance(data.get("vehicles"), list) else []

        if not results and not vehicles:
            return "Unknown", 0.0

        vehicle_info = None
        result_confidence = 0.0

        if results:
            result_confidence = float(results[0].get("confidence", 0) or 0)
            vehicle_info = results[0].get("vehicle")

        if not vehicle_info and vehicles:
            vehicle_info = vehicles[0].get("details")

        if not vehicle_info:
            return "Unknown", 0.0

        make = self._top_value(vehicle_info, "make")
        make_model = self._top_value(vehicle_info, "make_model")
        year_range = self._top_value(vehicle_info, "year")

        vehicle_confidence = self._top_confidence(vehicle_info, "make")

        if not make:
            return "Unknown", 0.0

        model = self._extract_model(make, make_model)
        year = self._parse_year(year_range)

        confidence = vehicle_confidence if vehicle_confidence > 0 else result_confidence

        return f"{year} {make.title()} {model}", confidence

    @staticmethod
    def _top_value(info: dict, key: str) -> str | None:
        items = info.get(key) if isinstance(info.get(key), list) else []

        if not items:
            return None

        return str(items[0].get("name", ""))

    @staticmethod
    def _top_confidence(info: dict, key: str) -> float:
        items = info.get(key) if isinstance(info.get(key), list) else []

        if not items:
            return 0.0

        return float(items[0].get("confidence", 0) or 0)

    @staticmethod
    def _extract_model(make: str, make_model: str | None) -> str:
        if not make_model:
            return "Unknown"

        prefix = make.lower().replace("-", " ").strip()
        combined = make_model.lower().replace("-", " ").strip()

        if combined.startswith(prefix):
            model = combined[len(prefix):].strip()
        else:
            model = combined

        model = model.replace("_", " ").replace("-", " ").strip()
        words = model.split()

        if words and words[-1].replace(".", "").isdigit():
            sep = words[-1]
            words = words[:-1]
            if words:
                return " ".join(words).title() + f" {sep}"

        return model.title() if model else "Unknown"

    @staticmethod
    def _parse_year(year_range: str | None) -> str:
        if not year_range:
            return "2020"

        import re
        match = re.match(r"(\d{4})-(\d{4})", year_range)

        if match:
            start, end = int(match.group(1)), int(match.group(2))
            return str(round((start + end) / 2))

        match = re.match(r"(\d{4})", year_range)

        if match:
            return match.group(1)

        return "2020"
