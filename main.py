import argparse
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2

from car_api import CarAPI
from fake_vehicle_identifier import FakeVehicleIdentifier
from geometry import (
    CAMERA_PRESETS,
    DEFAULT_CAMERA,
    GeometryEngine,
    format_distance,
    is_fully_visible,
)
from rekor_vehicle_identifier import RekorVehicleIdentifier
from vehicle_classifier import VehicleClassifier
from vehicle_detector import VehicleDetector


MEDIA_PATH = "demo_video.mp4"
ROTATE_VIDEO = True
CLASSIFY_EVERY = 20
MIN_CROP_WIDTH = 140
LIVE_READ_RETRIES = 30
LIVE_READ_RETRY_DELAY = 0.1
DEFAULT_RECORD_PATH = "metrics_recording.mp4"
DEFAULT_RECORD_FPS = 30.0
RECORD_BUTTON_WIDTH = 160
RECORD_BUTTON_HEIGHT = 42
RECORD_BUTTON_MARGIN = 18
CLASSIFY_WORKERS = 2

STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://")


def resolve_source(source: str) -> tuple[int | str, bool]:
    """
    Interpret a source string as a camera index, stream URL, or file.

    Returns (opencv_source, is_live).
    """
    cleaned = source.strip()

    if cleaned.isdigit():
        return int(cleaned), True

    if cleaned.lower().startswith(STREAM_PREFIXES):
        return cleaned, True

    media = Path(cleaned)

    if not media.exists():
        raise FileNotFoundError(f"{cleaned} was not found.")

    return str(media), False


def next_recording_path(base_path: str) -> str:
    path = Path(base_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if path.suffix:
        return str(path.with_name(f"{path.stem}_{timestamp}{path.suffix}"))

    return str(path / f"metrics_recording_{timestamp}.mp4")


def draw_record_button(frame, recording: bool) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    x2 = width - RECORD_BUTTON_MARGIN
    y1 = RECORD_BUTTON_MARGIN
    x1 = x2 - RECORD_BUTTON_WIDTH
    y2 = y1 + RECORD_BUTTON_HEIGHT

    color = (0, 0, 220) if recording else (60, 60, 60)
    label = "STOP REC" if recording else "START REC"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.circle(frame, (x1 + 22, y1 + 21), 8, (255, 255, 255), -1)
    cv2.putText(
        frame,
        label,
        (x1 + 42, y1 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return x1, y1, x2, y2


def open_recording_writer(
    video,
    frame,
    record_path: str,
) -> cv2.VideoWriter | None:
    height, width = frame.shape[:2]
    fps = video.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = DEFAULT_RECORD_FPS

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        record_path,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        writer.release()
        print(f"Could not start recording to {record_path}.")
        return None

    print(f"Recording output to {record_path}")
    return writer

# Representative trims for the MVP.
# These are used as default reference trims, not exact trim predictions.
DEFAULT_TRIMS = {
    ("toyota", "camry"): "LE",
    ("toyota", "corolla"): "LE",
    ("toyota", "rav4"): "LE",
    ("toyota", "highlander"): "LE",
    ("honda", "civic"): "LX",
    ("honda", "accord"): "Sport",
    ("honda", "cr-v"): "EX",
    ("honda", "pilot"): "EX-L",
    ("ford", "f-150"): "XLT",
    ("ford", "escape"): "SE",
    ("ford", "explorer"): "XLT",
    ("chevrolet", "silverado 1500"): "LT",
    ("chevrolet", "equinox"): "LT",
    ("nissan", "altima"): "SV",
    ("nissan", "sentra"): "SV",
    ("nissan", "rogue"): "SV",
    ("hyundai", "elantra"): "SEL",
    ("hyundai", "sonata"): "SEL",
    ("hyundai", "tucson"): "SEL",
    ("kia", "forte"): "LXS",
    ("kia", "optima"): "LX",
    ("kia", "sportage"): "LX",
    ("bmw", "3 series"): "330i",
    ("mercedes-benz", "c-class"): "C 300",
    ("tesla", "model 3"): "Long Range",
    ("tesla", "model y"): "Long Range",
}


def get_representative_trim(make: str, model: str) -> str | None:
    key = (make.lower().strip(), model.lower().strip())
    return DEFAULT_TRIMS.get(key)


def parse_vehicle_name(label: str):
    """
    Supports labels such as:
    2020 Toyota Camry
    2018-2022 Toyota Camry
    Toyota Camry 2020
    Honda CR-V 2019
    """

    cleaned = " ".join(label.strip().split())

    year_range_first = re.match(
        r"^(19\d{2}|20\d{2})-(19\d{2}|20\d{2})\s+"
        r"([A-Za-z][A-Za-z-]*)\s+(.+)$",
        cleaned,
    )

    if year_range_first:
        start_year, end_year, make, model = year_range_first.groups()
        year = round((int(start_year) + int(end_year)) / 2)
        return str(year), make, model

    year_first = re.match(
        r"^(19\d{2}|20\d{2})\s+([A-Za-z][A-Za-z-]*)\s+(.+)$",
        cleaned,
    )

    if year_first:
        year, make, model = year_first.groups()
        return year, make, model

    year_last = re.match(
        r"^([A-Za-z][A-Za-z-]*)\s+(.+?)\s+(19\d{2}|20\d{2})$",
        cleaned,
    )

    if year_last:
        make, model, year = year_last.groups()
        return year, make, model

    return None


def metric_lines(dimensions: dict | None) -> list[str]:
    if not dimensions:
        return ["Metrics unavailable"]

    lines = []

    trim = dimensions.get("trim")
    if trim:
        lines.append(f"Reference trim: {trim}")

    fields = [
        ("length", "Length", "in"),
        ("width", "Width", "in"),
        ("height", "Height", "in"),
        ("wheelbase", "Wheelbase", "in"),
        ("ground_clearance", "Clearance", "in"),
        ("curb_weight", "Weight", "lb"),
    ]

    for key, label, unit in fields:
        value = dimensions.get(key)

        if value not in (None, "", 0):
            lines.append(f"{label}: {value} {unit}")

    return lines or ["Metrics unavailable"]


def draw_label(frame, box, lines):
    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    line_height = 28
    start_y = max(35, y1 - line_height * len(lines) - 5)

    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x1, start_y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def annotate_frame(
    frame,
    frame_number,
    detector,
    classifier,
    car_api=None,
    geometry=None,
    prediction_cache=None,
    pending_classifications=None,
    last_classified_frame=None,
    executor=None,
    center_only=False,
    center_tolerance=0.25,
):
    """
    Run detection, tracking, classification, and metric overlays on one frame.

    Returns the annotated frame. Shared by the CLI player (main.py) and
    the REST API (api_server.py) so both consume the same pipeline.

    When classification is enabled (prediction_cache, pending_classifications,
    last_classified_frame, and executor are provided), tracked vehicles are
    classified asynchronously and metrics overlay after each result arrives.

    With center_only=True, only vehicles whose box center lies within
    center_tolerance (fraction of frame width) of the frame center get
    an overlay; everything else is skipped entirely.
    """
    if prediction_cache is None:
        prediction_cache = {}

    if pending_classifications is None:
        pending_classifications = {}

    if last_classified_frame is None:
        last_classified_frame = {}

    for tid in list(pending_classifications):
        future = pending_classifications[tid]

        if future.done():
            try:
                prediction_cache[tid] = future.result()
            except Exception as error:
                print(f"Classification error for track {tid}: {error}")
                prediction_cache[tid] = ("Unknown", 0.0, None, None)
            del pending_classifications[tid]

    result = detector.track(frame)

    for detection in result.boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())

        box = (x1, y1, x2, y2)
        class_id = int(detection.cls[0])
        detection_conf = float(detection.conf[0])
        box_width = x2 - x1

        if center_only:
            frame_width = frame.shape[1]
            box_center_x = (x1 + x2) / 2

            if abs(box_center_x - frame_width / 2) > frame_width * center_tolerance:
                continue

        track_id = (
            int(detection.id[0])
            if detection.id is not None
            else None
        )

        identity, identity_conf, trim, dimensions = (
            prediction_cache.get(
                track_id,
                ("Not classified", 0.0, None, None),
            )
        )

        should_classify = (
            executor is not None
            and track_id is not None
            and box_width >= MIN_CROP_WIDTH
            and track_id not in pending_classifications
            and (
                track_id not in prediction_cache
                or frame_number
                - last_classified_frame.get(track_id, -1)
                >= CLASSIFY_EVERY
            )
        )

        if should_classify:
            future = executor.submit(
                classify_and_get_metrics,
                frame=frame,
                box=box,
                classifier=classifier,
                car_api=car_api,
                track_id=track_id,
            )
            pending_classifications[track_id] = future
            last_classified_frame[track_id] = frame_number

        # Identified vehicles display make/model/year, trim,
        # and physical dimensions. Others show box only.
        lines = []

        if parse_vehicle_name(identity):
            lines.append(f"{identity}: {identity_conf:.1%}")

            if dimensions:
                lines.extend(metric_lines(dimensions))
            elif trim:
                lines.append(f"Reference trim: {trim}")

        # Geometry engine: estimate depth from the known vehicle
        # height and its pixel height in the current frame.
        if geometry is not None and dimensions:
            frame_height, frame_width = frame.shape[:2]

            if is_fully_visible(box, frame_width, frame_height):
                distance_m = geometry.estimate_distance_m(
                    real_size_in=dimensions.get("height"),
                    pixel_size=y2 - y1,
                    frame_width_px=frame_width,
                )

                if distance_m is not None:
                    lines.append(format_distance(distance_m))

        draw_label(frame, box, lines)

    cv2.putText(
        frame,
        f"Vehicles: {len(result.boxes)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    return frame


def classify_and_get_metrics(
    frame,
    box,
    classifier,
    car_api,
    track_id=None,
):
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]

    try:
        identity, identity_conf = classifier.classify(crop, track_id=track_id)
    except TypeError:
        identity, identity_conf = classifier.classify(crop)
    dimensions = None
    trim = None

    parsed_vehicle = parse_vehicle_name(identity)

    if parsed_vehicle:
        year, make, model = parsed_vehicle
        trim = get_representative_trim(make, model)

        print(
            f"Vehicle prediction: {year} {make} {model} "
            f"| Reference trim: {trim or 'API fallback'}"
        )

        dimensions = car_api.get_dimensions(
            year=year,
            make=make,
            model=model,
            trim=trim,
        )

    else:
        print(f"Could not parse classifier label: {identity}")

    return identity, identity_conf, trim, dimensions


def run_video(
    source: int | str,
    detector: VehicleDetector,
    classifier: VehicleClassifier,
    car_api: CarAPI | None = None,
    geometry: GeometryEngine | None = None,
    is_live: bool = False,
    rotate: bool = ROTATE_VIDEO,
    record_path: str = DEFAULT_RECORD_PATH,
    recording_enabled: bool = False,
):
    video = cv2.VideoCapture(source)

    if not video.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    if is_live:
        # Keep the buffer small so frames stay near real time.
        video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    prediction_cache = {}
    pending_classifications: dict[int, Future] = {}
    last_classified_frame: dict[int, int] = {}
    frame_number = 0
    failed_reads = 0
    writer = None
    active_record_path = record_path
    button_rect = None

    def handle_mouse(event, x, y, flags, param):
        nonlocal active_record_path, button_rect, recording_enabled, writer

        if event != cv2.EVENT_LBUTTONDOWN or button_rect is None:
            return

        x1, y1, x2, y2 = button_rect

        if not (x1 <= x <= x2 and y1 <= y <= y2):
            return

        if recording_enabled:
            recording_enabled = False

            if writer is not None:
                writer.release()
                writer = None
                print(f"Saved recording to {active_record_path}")

            return

        active_record_path = next_recording_path(record_path)
        recording_enabled = True

    cv2.namedWindow("MetricsAI", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("MetricsAI", handle_mouse)

    executor = ThreadPoolExecutor(max_workers=CLASSIFY_WORKERS)
    frame_times: list[float] = []

    while True:
        frame_start = time.perf_counter()
        success, frame = video.read()

        if not success:
            if is_live:
                failed_reads += 1

                if failed_reads > LIVE_READ_RETRIES:
                    print("Live feed lost. Stopping.")
                    break

                time.sleep(LIVE_READ_RETRY_DELAY)
                continue

            # File playback: loop back to the start.
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            prediction_cache.clear()
            frame_number = 0
            continue

        failed_reads = 0

        if rotate:
            frame = cv2.rotate(
                frame,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )

        annotate_frame(
            frame=frame,
            frame_number=frame_number,
            detector=detector,
            classifier=classifier,
            car_api=car_api,
            geometry=geometry,
            prediction_cache=prediction_cache,
            pending_classifications=pending_classifications,
            last_classified_frame=last_classified_frame,
            executor=executor,
        )

        button_rect = draw_record_button(frame, recording_enabled)

        if recording_enabled and writer is None:
            writer = open_recording_writer(
                video=video,
                frame=frame,
                record_path=active_record_path,
            )

            if writer is None:
                recording_enabled = False

        if writer is not None:
            writer.write(frame)

        frame_time = time.perf_counter() - frame_start
        frame_times.append(frame_time)

        if len(frame_times) > 30:
            frame_times.pop(0)

        avg_frame_time = sum(frame_times) / len(frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        cv2.putText(
            frame,
            f"FPS: {fps:.0f}",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )

        cv2.imshow("MetricsAI", frame)
        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    executor.shutdown(wait=False)

    video.release()

    if writer is not None:
        writer.release()
        print(f"Saved recording to {active_record_path}")

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MetricsAI vehicle metrics overlay.",
    )

    parser.add_argument(
        "--source",
        default=MEDIA_PATH,
        help=(
            "Video source: file path, camera index (e.g. 0), "
            "or stream URL (rtsp/http). "
            f"Defaults to {MEDIA_PATH}."
        ),
    )

    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        choices=sorted(CAMERA_PRESETS),
        help=(
            "Camera preset for depth estimation. "
            f"Defaults to {DEFAULT_CAMERA}."
        ),
    )

    parser.add_argument(
        "--focal-equiv",
        type=float,
        default=None,
        help=(
            "Override the 35mm-equivalent focal length in mm "
            "(iPhone 17 Pro Max main camera = 24.0)."
        ),
    )

    parser.add_argument(
        "--fake-identifier",
        action="store_true",
        help="Use fake vehicle identities for end-to-end metrics demos.",
    )

    parser.add_argument(
        "--rekor-identifier",
        action="store_true",
        help="Use Rekor CarCheck API for vehicle identification.",
    )

    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the processed MetricsAI output video.",
    )

    parser.add_argument(
        "--record-path",
        default=DEFAULT_RECORD_PATH,
        help=f"Output path for --record. Defaults to {DEFAULT_RECORD_PATH}.",
    )

    rotation = parser.add_mutually_exclusive_group()

    rotation.add_argument(
        "--rotate",
        dest="rotate",
        action="store_true",
        help="Rotate frames 90 degrees counterclockwise.",
    )

    rotation.add_argument(
        "--no-rotate",
        dest="rotate",
        action="store_false",
        help="Disable frame rotation.",
    )

    parser.set_defaults(rotate=None)

    return parser.parse_args()


def main():
    args = parse_args()
    source, is_live = resolve_source(str(args.source))

    # Default: rotate file playback (the demo clip needs it),
    # keep live feeds unrotated unless explicitly requested.
    rotate = args.rotate

    if rotate is None:
        rotate = ROTATE_VIDEO and not is_live

    focal_equiv = (
        args.focal_equiv
        if args.focal_equiv is not None
        else CAMERA_PRESETS[args.camera]
    )
    geometry = GeometryEngine(focal_equiv_mm=focal_equiv)
    print(
        f"Geometry engine ready: {args.camera} "
        f"({focal_equiv:.1f}mm equivalent)"
    )

    print("Authenticating with CarAPI...")
    car_api = CarAPI()
    car_api.authenticate()

    print("Loading computer-vision models...")
    detector = VehicleDetector()

    if args.fake_identifier:
        print("Using fake vehicle identifier for demo measurements.")
        classifier = FakeVehicleIdentifier()
    elif args.rekor_identifier:
        print("Using Rekor vehicle identifier.")
        classifier = RekorVehicleIdentifier()
    else:
        classifier = VehicleClassifier()

    feed_type = "live feed" if is_live else "video file"
    print(f"Starting MetricsAI on {feed_type}: {source}")

    if args.record:
        print(f"Recording enabled: {args.record_path}")

    run_video(
        source=source,
        detector=detector,
        classifier=classifier,
        car_api=car_api,
        geometry=geometry,
        is_live=is_live,
        rotate=rotate,
        record_path=args.record_path,
        recording_enabled=args.record,
    )


if __name__ == "__main__":
    main()
