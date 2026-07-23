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
