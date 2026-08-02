# MetricsMVP

MetricsMVP is a proof-of-concept for **MetricsAI**, a Physical Object Intelligence platform.

The MVP demonstrates that known real-world physical measurements can be attached to detected objects in real time. The current demo focuses on vehicles, but the long-term platform vision is a reusable physical-property layer for robotics, autonomy, fleet intelligence, and computer vision systems.

## Product Goal

Traditional computer vision answers:

```text
What is this object?
```

MetricsAI adds the next layer:

```text
What are this object's real-world physical properties?
```

For vehicles, this means attaching data such as:

- Length
- Width
- Height
- Wheelbase
- Ground clearance
- Curb weight
- Estimated distance from camera

## MVP Pipeline

```text
Video / Live Camera Feed
        ↓
OpenCV frame capture
        ↓
YOLO11 vehicle detection + tracking
        ↓
Vehicle crop extraction
        ↓
Vehicle identifier
        ↓
Representative trim selection
        ↓
CarAPI vehicle dimensions lookup
        ↓
Monocular geometry depth estimate
        ↓
Live overlay + optional recording
```

## Current Architecture

### `main.py`

Application orchestrator.

Responsibilities:

- Parse CLI arguments
- Open video files, webcam feeds, Continuity Camera, or stream URLs
- Rotate demo video when needed
- Run the frame loop
- Call YOLO tracking
- Crop detected vehicles
- Classify/identify each tracked vehicle
- Query CarAPI for vehicle measurements
- Estimate distance from camera
- Draw overlays
- Handle recording with an on-screen start/stop button

### `api_server.py`

REST API server wrapping the same pipeline.

Responsibilities:

- Accept video uploads (`POST /v1/process`) or file/stream URLs (`GET /v1/process`)
- Run the shared frame pipeline (`annotate_frame` in `main.py`)
- Return the overlaid video as an MP4 download

### `vehicle_detector.py`

YOLO11 wrapper.

Responsibilities:

- Load `yolo11s.pt`
- Detect vehicle classes only
- Track vehicles using Ultralytics/ByteTrack
- Return stable track IDs so identity and metrics stay attached to the same vehicle

Tracked COCO classes:

```text
car, motorcycle, bus, truck
```

### `vehicle_classifier.py`

Vehicle identity provider interface.

The current active implementation is Sighthound-compatible, but it requires credentials before it can return real make/model data.

Supported credential styles:

```env
SIGHTHOUND_API_KEY=...
```

or:

```env
SIGHTHOUND_ACCESS_TOKEN=...
```

The classifier preserves one internal interface:

```python
classify(crop) -> ("YYYY Make Model", confidence)
```

This keeps the rest of the pipeline independent from the specific vehicle-identification provider.

### `fake_vehicle_identifier.py`

Temporary demo identifier.

Used while waiting for production vehicle-ID credentials.

Responsibilities:

- Assign realistic fake vehicle identities
- Keep fake identities stable per tracked vehicle
- Allow the CarAPI + dimensions + geometry pipeline to be tested end-to-end

Run with:

```bash
python main.py --source demo_video.mp4 --fake-identifier
```

### `car_api.py`

Physical vehicle dimensions provider.

Responsibilities:

- Authenticate with CarAPI using `.env` credentials
- Query `/bodies/v2`
- Retrieve vehicle dimensions
- Cache lookups in memory
- Retry without trim if the representative trim fails
- Normalize classifier model names to CarAPI model names
- Clamp predicted years into the available CarAPI data range for MVP usage

Returned physical properties:

- Length
- Width
- Height
- Wheelbase
- Ground clearance
- Curb weight

### `geometry.py`

Monocular depth-estimation engine.

Responsibilities:

- Estimate vehicle distance from camera using the pinhole camera model
- Use known vehicle height from CarAPI
- Convert object pixel height into distance
- Support camera presets such as iPhone Continuity Camera

Core formula:

```text
distance = real_vehicle_height × focal_length_px / vehicle_pixel_height
```

Default camera preset:

```text
iPhone 17 Pro Max main camera, 24mm equivalent
```

## Runtime Modes

### Demo Video

```bash
python main.py --source demo_video.mp4
```

### Demo Video With Fake Vehicle IDs

Useful when no real vehicle identifier API key is available.

```bash
python main.py --source demo_video.mp4 --fake-identifier
```

### Live Camera / Continuity Camera

```bash
python main.py --source 1
```

Try other camera indexes if needed:

```bash
python main.py --source 0
python main.py --source 2
python main.py --source 3
```

### Record Output

```bash
python main.py --source demo_video.mp4 --fake-identifier --record
```

Or click the **START REC** / **STOP REC** button in the MetricsAI window.

### Override Camera Focal Length

```bash
python main.py --source 1 --focal-equiv 24
```

For 2x iPhone zoom:

```bash
python main.py --source 1 --focal-equiv 48
```

## Environment Variables

Create a `.env` file locally. Do not commit it.

Required for CarAPI dimensions:

```env
CARAPI_TOKEN=...
CARAPI_SECRET=...
```

Required for Sighthound vehicle identity when available:

```env
SIGHTHOUND_API_KEY=...
```

or:

```env
SIGHTHOUND_ACCESS_TOKEN=...
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

YOLO weights are not committed. Download or place the model file locally:

```text
yolo11s.pt
```

## REST API

`api_server.py` exposes the same pipeline over HTTP: send a video and get it back with the MetricsAI overlay baked in.

### Start the server

```bash
python api_server.py --identifier fake --port 8000
```

Flags:

- `--identifier fake|sighthound|rekor` — vehicle identity provider. `fake` needs no credentials and is the default demo mode; the others require their `.env` keys.
- `--camera <preset>` / `--focal-equiv <mm>` — depth-estimation settings (same as `main.py`).
- `--host`, `--port` — listen address, default `0.0.0.0:8000`.

### Upload a video file

```bash
curl -X POST http://127.0.0.1:8000/v1/process \
  -F "video=@demo_video.mp4" \
  -o overlay.mp4
```

### Process a local file or stream URL

```bash
# local path on the server
curl "http://127.0.0.1:8000/v1/process?url=/abs/path/demo_video.mp4" -o overlay.mp4

# live RTSP / HTTP stream (cap with max_seconds; streams may never end)
curl "http://127.0.0.1:8000/v1/process?url=rtsp://127.0.0.1:8554/la_demo&max_seconds=30" -o overlay.mp4
```

### Query parameters

| Param             | Applies to | Meaning                                                     |
| ----------------- | ---------- | ----------------------------------------------------------- |
| `rotate`          | both       | `auto` (default) rotates portrait videos (height > width) to landscape; `true` always rotates; `false` never rotates. |
| `max_frames`      | both       | Process at most N frames.                                   |
| `max_seconds`     | both       | Process at most N seconds of video content (video time, not wall time). |
| `max_wall_seconds`| both       | Process at most N seconds of real time (processing is slower than real time on CPU). |
| `center_only`     | both       | Overlay only vehicles near the frame center (`true`) — the ones in front of the camera. |
| `center_tolerance`| both       | Max horizontal offset from frame center, as a fraction of frame width (default `0.15`, i.e. a 30%-wide center window; range `0`–`0.5`). Used with `center_only`. |

Response headers: `X-Processed-Frames`, `X-Frame-Rate`, `X-Resolution`.

### Other endpoints

- `GET /health` — liveness check (models load lazily on first processing request).
- `GET /` — endpoint listing.

### Testing the API

Start the server, then run these checks:

```bash
python api_server.py --identifier fake --port 8000
```

1. Health check:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","pipeline_loaded":false,"identifier":"fake"}
```

2. Quick smoke test with the small demo clip (returns in seconds):

```bash
# auto-rotation: demo_video.mp4 is portrait and is rotated automatically
curl "http://127.0.0.1:8000/v1/process?url=$(pwd)/demo_video.mp4&max_frames=10" \
  -o /tmp/smoke.mp4

# Center-only variant: overlay just the vehicles ahead of the camera
curl "http://127.0.0.1:8000/v1/process?url=$(pwd)/demo_video.mp4&center_only=true&max_frames=10" \
  -o /tmp/smoke_center.mp4

# Expect: HTTP 200, X-Processed-Frames: 10, X-Frame-Rate: ~33
# The 10-frame output should be ~0.3s long and show vehicle boxes + overlay:
ffprobe -v error -show_entries format=duration -of csv=p=0 /tmp/smoke.mp4
ffplay -autoexit /tmp/smoke.mp4
```

3. Process a clip of the large 4K demo feed:

```bash
curl "http://127.0.0.1:8000/v1/process?url=$(pwd)/la_demo.mp4&max_seconds=10" \
  -o /tmp/la_demo_10s.mp4
```

`max_seconds` is content time, not wall time: 10 s of 4K video is ~300
frames and takes ~20 s of wall time on CPU, with progress logged to the
server console every 100 frames. Verify the output keeps the source speed:

```bash
ffprobe -v error -show_entries format=duration -of csv=p=0 /tmp/la_demo_10s.mp4   # ~10.0 s
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,r_frame_rate -of csv=p=0 /tmp/la_demo_10s.mp4
# h264, ~29.97 fps
```

4. RTSP live-stream test (optional; requires `tools/stream_file.py`):

```bash
tools/stream_file.py la_demo.mp4
curl "http://127.0.0.1:8000/v1/process?url=rtsp://127.0.0.1:8554/la_demo&max_seconds=10&max_wall_seconds=30" \
  -o /tmp/rtsp_test.mp4
```

Notes:

- Processing is synchronous; a large video returns once all frames are overlaid.
- The shared YOLO model serializes concurrent processing requests.
- Portrait clips (e.g. `demo_video.mp4`) are auto-rotated to landscape by default.
- Output is H.264 in an MP4 container, encoded via ffmpeg with exact
  timestamps, so playback speed always matches the source. If ffmpeg is
  not installed, the server falls back to OpenCV's mp4v writer (known to
  play too fast in some players at fractional frame rates).
- `max_seconds` caps content time: 30 s of 4K video at ~30 fps is ~900
  frames and takes minutes of wall time on CPU — pair it with
  `max_wall_seconds` for a real-time bound, or use `max_frames` for
  quick checks.

## Current MVP Limitations

- Vehicle trim is not predicted directly. The MVP uses representative trims.
- Distance estimation assumes the selected camera focal length is correct.
- Center Stage, Portrait Mode, and digital zoom can change effective focal length.
- Monocular depth is an estimate, not lidar-grade measurement.
- Real vehicle identification depends on the external provider credential being available.
- The fake identifier mode is only for pipeline demos and should not be used as real recognition.

## Long-Term Platform Direction

The vehicle demo is the first proof-of-concept for a broader Physical Object Intelligence platform.

Future object categories may include:

- Street signs
- Traffic lights
- Trash cans
- Road barriers
- Construction equipment
- Utility poles
- Sidewalk infrastructure
- Public infrastructure assets

The long-term goal is a developer API that turns detected objects into structured physical intelligence for autonomous systems.
