"""
MetricsAI REST API.

Ingest a video file (multipart upload) or a stream URL, run the same
detection / classification / metrics pipeline as main.py, and return the
video with the OpenCV overlay baked in.

Endpoints:

    POST /v1/process   Upload a video file -> overlaid MP4
    GET  /v1/process   ?url=<path|rtsp|http|...> -> overlaid MP4
    GET  /health       Liveness check

Run:

    python api_server.py --identifier fake --port 8000
"""

import argparse
import logging
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import cv2
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from car_api import CarAPI
from fake_vehicle_identifier import FakeVehicleIdentifier
from geometry import CAMERA_PRESETS, DEFAULT_CAMERA, GeometryEngine
from main import (
    CLASSIFY_WORKERS,
    DEFAULT_RECORD_FPS,
    annotate_frame,
)
from rekor_vehicle_identifier import RekorVehicleIdentifier
from vehicle_classifier import VehicleClassifier
from vehicle_detector import VehicleDetector

load_dotenv()

app = FastAPI(
    title="MetricsAI API",
    description=(
        "Upload a video or point at a stream URL and get it back with the "
        "MetricsAI vehicle detection / metrics overlay."
    ),
    version="1.0.0",
)

logger = logging.getLogger("metricsai.api")

ALLOWED_URL_SCHEMES = {"http", "https", "rtsp", "rtmp", "udp", "file"}
OPEN_TIMEOUT_MS = 10_000
PROGRESS_EVERY = 100

# Pipeline configuration from the command line; the models themselves are
# loaded lazily on the first request so /health works immediately.
CONFIG = {
    "identifier": "fake",
    "camera": DEFAULT_CAMERA,
    "focal_equiv": None,
}

_state_lock = threading.Lock()
_pipeline: "PipelineState | None" = None
_process_lock = threading.Lock()


def build_classifier(identifier: str):
    if identifier == "rekor":
        return RekorVehicleIdentifier()

    if identifier == "sighthound":
        classifier = VehicleClassifier()

        if classifier.mode == "disabled":
            logger.warning(
                "Sighthound credentials not found; "
                "classifier will return 'Unknown' identities."
            )

        return classifier

    if identifier == "fake":
        return FakeVehicleIdentifier()

    raise ValueError(f"Unknown identifier: {identifier}")


class PipelineState:
    """Shared, lazily-loaded computer vision pipeline."""

    def __init__(self, identifier: str, camera: str, focal_equiv: float | None):
        self.identifier = identifier
        self.camera = camera
        self.focal_equiv = focal_equiv or CAMERA_PRESETS[camera]
        self.geometry = GeometryEngine(focal_equiv_mm=self.focal_equiv)
        self.detector = VehicleDetector()
        self.classifier = build_classifier(identifier)
        self.car_api = CarAPI()
        self.car_api.authenticate()


def get_pipeline() -> PipelineState:
    global _pipeline

    with _state_lock:
        if _pipeline is None:
            logger.info(
                "Loading pipeline: identifier=%s camera=%s "
                "focal_equiv=%s",
                CONFIG["identifier"],
                CONFIG["camera"],
                CONFIG["focal_equiv"],
            )
            _pipeline = PipelineState(
                identifier=CONFIG["identifier"],
                camera=CONFIG["camera"],
                focal_equiv=CONFIG["focal_equiv"],
            )

        return _pipeline


def rotate_frame(frame) -> Any:
    return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


def should_rotate(rotate_mode: str, frame) -> bool:
    """
    Resolve the rotate option for a frame.

    "true" always rotates, "false" never does, and "auto" rotates
    portrait frames (height > width) so phone/dashcam clips display
    as landscape.
    """
    if rotate_mode == "true":
        return True

    if rotate_mode == "false":
        return False

    height, width = frame.shape[:2]
    return height > width


class FfmpegVideoWriter:
    """
    Stream annotated frames into ffmpeg for H.264 encoding.

    OpenCV's mp4v writer mangles fractional frame rates (e.g. 29.97),
    which makes some players run the output too fast. ffmpeg stores
    exact 30000/1001-style timestamps, so the output plays at the
    source speed.
    """

    def __init__(self, output_path: str, width: int, height: int, fps: float):
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        self.frames_written = 0

    def write(self, frame) -> None:
        try:
            self.process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as error:
            raise ValueError(f"ffmpeg encoder failed: {error}") from error

        self.frames_written += 1

        if self.process.poll() is not None:
            raise ValueError("ffmpeg encoder exited before the video finished.")

    def finish(self) -> None:
        """Close the pipe and wait for ffmpeg to finalize the file."""
        if self.process.stdin is not None:
            self.process.stdin.close()

        returncode = self.process.wait()

        if returncode != 0:
            raise ValueError(f"ffmpeg exited with code {returncode}.")

    def abort(self) -> None:
        """Best-effort shutdown that never raises."""
        if self.process.poll() is not None:
            return

        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        except OSError:
            pass

        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


class _Cv2VideoWriter:
    """Fallback when ffmpeg is not installed."""

    def __init__(self, output_path: str, width: int, height: int, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height),
        )

        if not self.writer.isOpened():
            raise ValueError(f"Could not start video writer for {output_path}.")

    def write(self, frame) -> None:
        self.writer.write(frame)

    def finish(self) -> None:
        self.writer.release()

    def abort(self) -> None:
        self.writer.release()


def open_video_writer(output_path: str, width: int, height: int, fps: float):
    if shutil.which("ffmpeg"):
        return FfmpegVideoWriter(output_path, width, height, fps)

    logger.warning(
        "ffmpeg not found on PATH; falling back to the OpenCV mp4v "
        "writer, which may play too fast in some players."
    )
    return _Cv2VideoWriter(output_path, width, height, fps)


def process_video_source(
    source: int | str,
    output_path: str,
    *,
    rotate: str = "auto",
    max_frames: int | None,
    max_seconds: float | None,
    max_wall_seconds: float | None,
    center_only: bool = False,
    center_tolerance: float = 0.15,
) -> dict[str, Any]:
    """
    Run the MetricsAI overlay pipeline over a video source.

    The shared YOLO model serializes processing per request for stability.
    """
    pipeline = get_pipeline()
    video = cv2.VideoCapture(source)

    if not video.isOpened():
        raise ValueError(f"Could not open video source: {source}")

    if isinstance(source, str):
        video.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS)

    success, frame = video.read()

    if not success:
        video.release()
        raise ValueError("Video source produced no readable frames.")

    rotate_frames = should_rotate(rotate, frame)

    if rotate_frames:
        frame = rotate_frame(frame)

    height, width = frame.shape[:2]
    fps = video.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0 or fps > 120:
        fps = DEFAULT_RECORD_FPS

    writer = open_video_writer(output_path, width, height, fps)
    executor = ThreadPoolExecutor(max_workers=CLASSIFY_WORKERS)
    prediction_cache: dict = {}
    pending_classifications: dict = {}
    last_classified_frame: dict = {}
    frame_number = 0
    started = time.perf_counter()

    try:
        with _process_lock:
            while frame is not None:
                frame = annotate_frame(
                    frame=frame,
                    frame_number=frame_number,
                    detector=pipeline.detector,
                    classifier=pipeline.classifier,
                    car_api=pipeline.car_api,
                    geometry=pipeline.geometry,
                    prediction_cache=prediction_cache,
                    pending_classifications=pending_classifications,
                    last_classified_frame=last_classified_frame,
                    executor=executor,
                    center_only=center_only,
                    center_tolerance=center_tolerance,
                )

                writer.write(frame)
                frame_number += 1

                if max_frames and frame_number >= max_frames:
                    break

                if max_seconds and frame_number / fps >= max_seconds:
                    break

                if (
                    max_wall_seconds
                    and time.perf_counter() - started >= max_wall_seconds
                ):
                    break

                if frame_number % PROGRESS_EVERY == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"Processed {frame_number} frames "
                        f"({frame_number / fps:.1f}s of content, "
                        f"{elapsed:.0f}s wall).",
                        flush=True,
                    )

                success, frame = video.read()

                if not success:
                    break

                if rotate_frames:
                    frame = rotate_frame(frame)

        writer.finish()
    finally:
        executor.shutdown(wait=False)
        video.release()
        writer.abort()

    if frame_number == 0:
        raise ValueError("Video source produced no frames.")

    elapsed = time.perf_counter() - started
    print(
        f"Done: {frame_number} frames, {frame_number / fps:.1f}s of content "
        f"in {elapsed:.0f}s wall time -> {output_path}",
        flush=True,
    )

    return {
        "frames": frame_number,
        "fps": fps,
        "width": width,
        "height": height,
    }


def resolve_url_source(url: str) -> str:
    """Accept a local file path or a whitelisted stream URL."""
    cleaned = url.strip()

    if "://" in cleaned:
        scheme = cleaned.split("://", 1)[0].lower()

        if scheme not in ALLOWED_URL_SCHEMES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported URL scheme: {scheme}",
            )

        return cleaned

    path = Path(cleaned).expanduser()

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {cleaned}",
        )

    return str(path)


def overlay_response(
    background_tasks: BackgroundTasks,
    stats: dict[str, Any],
    output_path: str,
) -> FileResponse:
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename="metricsai_overlay.mp4",
        headers={
            "X-Processed-Frames": str(stats["frames"]),
            "X-Frame-Rate": f"{stats['fps']:.2f}",
            "X-Resolution": f"{stats['width']}x{stats['height']}",
        },
        background=background_tasks,
    )


@app.get("/")
def index() -> dict:
    return {
        "service": "MetricsAI API",
        "version": "1.0.0",
        "endpoints": {
            "POST /v1/process": (
                "multipart upload of a video file; "
                "returns it with the overlay"
            ),
            "GET /v1/process": (
                "query param url=... (local path or http/rtsp stream); "
                "returns the source with the overlay"
            ),
            "GET /health": "liveness check",
        },
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "pipeline_loaded": _pipeline is not None,
        "identifier": CONFIG["identifier"],
    }


@app.post("/v1/process")
def process_upload(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Video file to process"),
    rotate: Literal["auto", "true", "false"] = Query(
        default="auto",
        description=(
            "auto rotates portrait videos (height > width) to landscape; "
            "true always rotates; false never rotates."
        ),
    ),
    max_frames: int = Query(
        default=None,
        ge=1,
        description="Process at most this many frames.",
    ),
    max_seconds: float = Query(
        default=None,
        gt=0,
        description="Process at most this many seconds of video.",
    ),
    max_wall_seconds: float = Query(
        default=None,
        gt=0,
        description=(
            "Process at most this many seconds of real time "
            "(processing is slower than real time on CPU)."
        ),
    ),
    center_only: bool = Query(
        default=False,
        description=(
            "Overlay only vehicles near the frame center "
            "(the ones in front of the camera)."
        ),
    ),
    center_tolerance: float = Query(
        default=0.15,
        ge=0,
        le=0.5,
        description=(
            "Max horizontal offset from the frame center, as a "
            "fraction of frame width (default 0.15 = a 30% wide "
            "center window), for center_only vehicles."
        ),
    ),
):
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    workdir = Path(tempfile.mkdtemp(prefix="metricsai_upload_"))
    input_path = workdir / f"input{suffix}"
    output_path = workdir / "metricsai_overlay.mp4"

    with input_path.open("wb") as out:
        while chunk := video.file.read(1 << 20):
            out.write(chunk)

    background_tasks.add_task(shutil.rmtree, workdir, ignore_errors=True)

    try:
        stats = process_video_source(
            source=str(input_path),
            output_path=str(output_path),
            rotate=rotate,
            max_frames=max_frames,
            max_seconds=max_seconds,
            max_wall_seconds=max_wall_seconds,
            center_only=center_only,
            center_tolerance=center_tolerance,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Failed to process uploaded video.")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {error}",
        ) from error

    return overlay_response(background_tasks, stats, str(output_path))


@app.get("/v1/process")
def process_url(
    background_tasks: BackgroundTasks,
    url: str = Query(
        ...,
        description=(
            "Local file path or stream URL "
            "(http, https, rtsp, rtmp, udp, file)."
        ),
    ),
    rotate: Literal["auto", "true", "false"] = Query(
        default="auto",
        description=(
            "auto rotates portrait videos (height > width) to landscape; "
            "true always rotates; false never rotates."
        ),
    ),
    max_frames: int = Query(
        default=None,
        ge=1,
        description="Process at most this many frames.",
    ),
    max_seconds: float = Query(
        default=None,
        gt=0,
        description="Process at most this many seconds of stream.",
    ),
    max_wall_seconds: float = Query(
        default=None,
        gt=0,
        description=(
            "Process at most this many seconds of real time "
            "(processing is slower than real time on CPU)."
        ),
    ),
    center_only: bool = Query(
        default=False,
        description=(
            "Overlay only vehicles near the frame center "
            "(the ones in front of the camera)."
        ),
    ),
    center_tolerance: float = Query(
        default=0.15,
        ge=0,
        le=0.5,
        description=(
            "Max horizontal offset from the frame center, as a "
            "fraction of frame width (default 0.15 = a 30% wide "
            "center window), for center_only vehicles."
        ),
    ),
):
    source = resolve_url_source(url)
    workdir = Path(tempfile.mkdtemp(prefix="metricsai_url_"))
    output_path = workdir / "metricsai_overlay.mp4"
    background_tasks.add_task(shutil.rmtree, workdir, ignore_errors=True)

    try:
        stats = process_video_source(
            source=source,
            output_path=str(output_path),
            rotate=rotate,
            max_frames=max_frames,
            max_seconds=max_seconds,
            max_wall_seconds=max_wall_seconds,
            center_only=center_only,
            center_tolerance=center_tolerance,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Failed to process stream source %s.", url)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {error}",
        ) from error

    return overlay_response(background_tasks, stats, str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MetricsAI REST API server.",
    )

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--identifier",
        choices=("fake", "sighthound", "rekor"),
        default=CONFIG["identifier"],
        help=(
            "Vehicle identity provider. fake uses demo identities; "
            "sighthound and rekor use their respective APIs "
            "(requires .env credentials). Defaults to fake."
        ),
    )
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        choices=sorted(CAMERA_PRESETS),
        help="Camera preset for depth estimation.",
    )
    parser.add_argument(
        "--focal-equiv",
        type=float,
        default=None,
        help="Override the 35mm-equivalent focal length in mm.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CONFIG.update(
        identifier=args.identifier,
        camera=args.camera,
        focal_equiv=args.focal_equiv,
    )
    print(f"MetricsAI API listening on http://{args.host}:{args.port}")
    print(
        f"Pipeline config: identifier={args.identifier} "
        f"camera={args.camera} focal_equiv={args.focal_equiv or 'default'}"
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
