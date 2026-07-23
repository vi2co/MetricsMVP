import base64
import os
from typing import Any

import cv2
import requests
from dotenv import load_dotenv


load_dotenv()


class VehicleClassifier:
    """
    Vehicle make/model identifier backed by Sighthound.

    Supports either:
    - v11 images:annotate endpoint using SIGHTHOUND_API_KEY
    - v1 recognition endpoint using SIGHTHOUND_ACCESS_TOKEN

    The public interface remains compatible with the rest of MetricsMVP:
        classify(crop) -> ("YYYY Make Model", confidence)
    """

    V11_URL = "https://preview.sighthoundapi.com/v11/images:annotate"
    V1_URL = "https://dev.sighthoundapi.com/v1/recognition?objectType=vehicle"

    def __init__(self):
        self.api_key = os.getenv("SIGHTHOUND_API_KEY", "").strip()
        self.access_token = os.getenv("SIGHTHOUND_ACCESS_TOKEN", "").strip()
        self.api_url = os.getenv("SIGHTHOUND_API_URL", "").strip()

        if self.api_url:
            self.mode = "v11" if "images:annotate" in self.api_url else "v1"
        elif self.api_key:
            self.api_url = self.V11_URL
            self.mode = "v11"
        elif self.access_token:
            self.api_url = self.V1_URL
            self.mode = "v1"
        else:
            self.mode = "disabled"
            print(
                "Sighthound credentials not found. Add SIGHTHOUND_API_KEY "
                "or SIGHTHOUND_ACCESS_TOKEN to .env."
            )

    def classify(self, crop):
        """
        Input:
            crop -> OpenCV image (numpy array)

        Returns:
            ("vehicle label", confidence)
        """
        if crop is None or crop.size == 0:
            return "Unknown", 0.0

        if self.mode == "disabled":
            return "Unknown", 0.0

        success, buffer = cv2.imencode(".jpg", crop)

        if not success:
            print("Failed to encode vehicle crop.")
            return "Unknown", 0.0

        if self.mode == "v1":
            return self._classify_v1(buffer.tobytes())

        return self._classify_v11(buffer.tobytes())

    def _classify_v11(self, image_bytes: bytes) -> tuple[str, float]:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        request_body = {
            "requests": [
                {
                    "image": {"content": image_b64},
                    "features": [{"type": "VEHICLE_DETECTION"}],
                }
            ]
        }
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["X-API-Key"] = self.api_key

        data = self._post_json(request_body, headers)

        if not data:
            return "Unknown", 0.0

        vehicles = data.get("responses", [{}])[0].get("vehicleAnnotations", [])

        if not vehicles:
            return "Unknown", 0.0

        vehicle = max(vehicles, key=lambda item: item.get("score", 0))
        return self._format_v11_vehicle(vehicle)

    def _classify_v1(self, image_bytes: bytes) -> tuple[str, float]:
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Access-Token": self.access_token,
        }
        data = self._post_bytes(image_bytes, headers)

        if not data:
            return "Unknown", 0.0

        objects = [
            item
            for item in data.get("objects", [])
            if item.get("objectType") == "vehicle"
        ]

        if not objects:
            return "Unknown", 0.0

        vehicle = max(
            objects,
            key=lambda item: item.get("vehicleAnnotation", {}).get(
                "recognitionConfidence",
                0,
            ),
        )
        return self._format_v1_vehicle(vehicle)

    def _post_json(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any] | None:
        try:
            response = requests.post(
                self.api_url,
                json=body,
                headers=headers,
                timeout=20,
            )

            if response.status_code != 200:
                print(f"Sighthound API error ({response.status_code}): {response.text}")
                return None

            return response.json()

        except requests.Timeout:
            print("Sighthound API request timed out.")
        except requests.RequestException as error:
            print(f"Sighthound API request failed: {error}")
        except ValueError:
            print("Sighthound API returned invalid JSON.")

        return None

    def _post_bytes(
        self,
        image_bytes: bytes,
        headers: dict[str, str],
    ) -> dict[str, Any] | None:
        try:
            response = requests.post(
                self.api_url,
                data=image_bytes,
                headers=headers,
                timeout=20,
            )

            if response.status_code != 200:
                print(f"Sighthound API error ({response.status_code}): {response.text}")
                return None

            return response.json()

        except requests.Timeout:
            print("Sighthound API request timed out.")
        except requests.RequestException as error:
            print(f"Sighthound API request failed: {error}")
        except ValueError:
            print("Sighthound API returned invalid JSON.")

        return None

    @staticmethod
    def _format_v11_vehicle(vehicle: dict[str, Any]) -> tuple[str, float]:
        make = vehicle.get("make", {}).get("value")
        model = vehicle.get("model", {}).get("value")
        generation = vehicle.get("generation", {}).get("value", {})
        confidence = float(vehicle.get("score", 0) or 0)

        year = _representative_year(
            generation.get("start"),
            generation.get("end"),
        )
        label = _label_from_parts(year, make, model)
        return label, confidence

    @staticmethod
    def _format_v1_vehicle(vehicle: dict[str, Any]) -> tuple[str, float]:
        annotation = vehicle.get("vehicleAnnotation", {})
        attributes = annotation.get("attributes", {}).get("system", {})
        make = attributes.get("make", {}).get("name")
        model = attributes.get("model", {}).get("name")
        confidence = float(annotation.get("recognitionConfidence", 0) or 0)

        # v1 does not include generation years in the documented response.
        # Use a CarAPI-supported representative year so dimensions still work.
        label = _label_from_parts("2020", make, model)
        return label, confidence


def _representative_year(start: Any, end: Any) -> str:
    try:
        start_year = int(start)
        end_year = int(end) if end else start_year
    except (TypeError, ValueError):
        return "2020"

    return str(round((start_year + end_year) / 2))


def _label_from_parts(year: str, make: str | None, model: str | None) -> str:
    parts = [year]

    if make:
        parts.append(str(make))

    if model:
        parts.append(str(model))

    return " ".join(parts) if len(parts) > 1 else "Unknown"
