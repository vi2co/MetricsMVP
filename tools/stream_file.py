#!/usr/bin/env python3
"""
Stream a local mp4/webm video file as a live RTSP feed.

Publishes the file into the bundled MediaMTX server with ffmpeg, so the
MetricsAI pipeline (or any RTSP client) can consume it as a live stream.

    tools/stream_file.py                     # streams la_demo.webm
    tools/stream_file.py demo_video.mp4      # streams another file
    tools/stream_file.py --no-loop clip.mp4  # play once, then stop

Consume with:

    python main.py --source rtsp://127.0.0.1:8554/la_demo
"""

import argparse
import json
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MEDIA = "la_demo.mp4"
DEFAULT_RTSP_PORT = 8554
STREAM_LOOP = True
KEYFRAME_INTERVAL = 48

TOOLS_DIR = Path(__file__).resolve().parent
MEDIAMTX_BIN = TOOLS_DIR / "mediamtx"
MEDIAMTX_CONFIG = TOOLS_DIR / "mediamtx.yml"
REPO_DIR = TOOLS_DIR.parent


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def probe_stream(path: str) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout).get("streams", [])
    return streams[0] if streams else {}


def wait_for_port(port: int, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_open(port):
            return True
        time.sleep(0.25)
    return False


def start_mediamtx(port: int) -> subprocess.Popen | None:
    if port_open(port):
        print(f"MediaMTX already running on port {port}.")
        return None

    if not MEDIAMTX_BIN.is_file():
        print(
            f"MediaMTX binary not found at {MEDIAMTX_BIN}. "
            "Download it from https://github.com/bluenviron/mediamtx/releases."
        )
        return None

    print(f"Starting MediaMTX server (port {port})...")
    process = subprocess.Popen(
        [str(MEDIAMTX_BIN), str(MEDIAMTX_CONFIG)],
        cwd=TOOLS_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not wait_for_port(port):
        process.terminate()
        print("MediaMTX failed to start.")
        return None

    print("MediaMTX is ready.")
    return process


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name.strip()) or "stream"


def resolve_media(path: str) -> str:
    candidate = Path(path)

    if candidate.is_file():
        return str(candidate.resolve())

    alt = REPO_DIR / path

    if alt.is_file():
        return str(alt.resolve())

    raise FileNotFoundError(f"Media file not found: {path}")


def build_ffmpeg_command(
    media: str,
    name: str,
    port: int,
    loop: bool,
    transcode: bool,
    scale: str,
) -> list[str]:
    stream = probe_stream(media)
    codec = stream.get("codec_name", "")
    width = int(stream.get("width", 0) or 0)
    height = int(stream.get("height", 0) or 0)

    needs_transcode = transcode or codec != "h264"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
    ]

    if loop:
        command += ["-stream_loop", "-1"]

    command += ["-re", "-i", media, "-an"]

    if needs_transcode:
        print(
            f"Codec {codec or 'unknown'} requires transcoding to H.264 "
            "for RTSP compatibility."
        )
        command += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(KEYFRAME_INTERVAL),
        ]

        if scale == "auto" and (width > 1920 or height > 1080):
            print(f"Downscaling {width}x{height} to 1920x1080 for realtime encoding.")
            command += ["-vf", "scale=1920:1080"]
        elif scale == "1080":
            command += ["-vf", "scale=1920:1080"]
        elif scale == "720":
            command += ["-vf", "scale=1280:720"]
    else:
        print(f"Streaming {codec.upper()} without re-encoding.")
        command += ["-c:v", "copy"]

    command += [
        "-f",
        "rtsp",
        f"rtsp://127.0.0.1:{port}/{name}",
    ]
    return command


def lan_addresses() -> list[str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return [sock.getsockname()[0]]
    except OSError:
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream an mp4/webm file as a live RTSP feed.",
    )

    parser.add_argument(
        "file",
        nargs="?",
        default=DEFAULT_MEDIA,
        help=f"Video file to stream. Defaults to {DEFAULT_MEDIA}.",
    )

    parser.add_argument(
        "--name",
        default=None,
        help="RTSP stream name. Defaults to the file name without extension.",
    )

    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Play the file once and stop instead of looping forever.",
    )

    parser.add_argument(
        "--transcode",
        action="store_true",
        help="Force H.264 re-encoding even for H.264 sources.",
    )

    parser.add_argument(
        "--scale",
        choices=("auto", "original", "1080", "720"),
        default="auto",
        help=(
            "Output resolution when transcoding. "
            "auto keeps the source size unless it exceeds 1080p."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_RTSP_PORT,
        help=f"RTSP port (fixed at {DEFAULT_RTSP_PORT} due to bundled MediaMTX config).",
    )

    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Assume MediaMTX is already running on port 8554; do not start or stop it.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    media = resolve_media(args.file)
    name = args.name or Path(media).stem
    name = sanitize_name(name)

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH. Install it to use this tool.")
        sys.exit(1)

    if args.port != DEFAULT_RTSP_PORT:
        print(
            f"RTSP port {args.port} is not supported by the bundled MediaMTX config. "
            f"Use port {DEFAULT_RTSP_PORT}."
        )
        sys.exit(1)

    server = None if args.no_server else start_mediamtx(args.port)

    if server is None and not port_open(args.port):
        print(f"RTSP server is not reachable on port {args.port}. Aborting.")
        sys.exit(1)

    command = build_ffmpeg_command(
        media=media,
        name=name,
        port=args.port,
        loop=not args.no_loop,
        transcode=args.transcode,
        scale=args.scale,
    )

    print(f"Streaming {media} -> rtsp://127.0.0.1:{args.port}/{name}", flush=True)

    for address in lan_addresses():
        print(f"LAN clients:      rtsp://{address}:{args.port}/{name}", flush=True)

    print(
        f"MetricsAI:        python main.py --source rtsp://127.0.0.1:{args.port}/{name}",
        flush=True,
    )
    print("Press Ctrl+C to stop.", flush=True)

    process = subprocess.Popen(command)

    def stop_server(_signum, _frame):
        process.terminate()

    signal.signal(signal.SIGTERM, stop_server)

    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
    finally:
        if server is not None:
            print("Stopping MediaMTX...")
            server.terminate()
            server.wait()


if __name__ == "__main__":
    main()
