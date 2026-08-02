"""
Monocular geometry engine for MetricsAI.

Estimates real-world distance to detected vehicles using the pinhole
camera model. Because the platform knows each vehicle's verified
physical dimensions, depth can be recovered from a single camera:

    distance = real_size * focal_length_px / pixel_size

The focal length in pixels is derived from the camera's 35mm-equivalent
focal length and the delivered frame width:

    focal_length_px = frame_width_px * focal_equiv_mm / 36.0

Vehicle height is used as the reference dimension because a vehicle's
pixel height is nearly viewpoint-invariant, while pixel width varies
about 2x between frontal and side views.
"""

INCHES_PER_METER = 39.3701
FULL_FRAME_WIDTH_MM = 36.0  # 35mm-format reference width

# 35mm-equivalent focal lengths for supported cameras.
# Continuity Camera uses the main (wide) lens by default.
CAMERA_PRESETS = {
    "iphone-17-pro-max": 24.0,  # main 48MP Fusion camera, 24mm equiv
    "iphone-17-pro-max-2x": 48.0,  # 2x sensor-crop zoom
    "generic-webcam": 26.0,
}

DEFAULT_CAMERA = "iphone-17-pro-max"


class GeometryEngine:
    """
    Converts known physical dimensions + pixel measurements into
    real-world distance estimates.
    """

    def __init__(self, focal_equiv_mm: float = CAMERA_PRESETS[DEFAULT_CAMERA]):
        if focal_equiv_mm <= 0:
            raise ValueError("focal_equiv_mm must be positive.")

        self.focal_equiv_mm = focal_equiv_mm

    def focal_length_px(self, frame_width_px: int) -> float:
        """Focal length in pixels for the delivered frame width."""
        return frame_width_px * self.focal_equiv_mm / FULL_FRAME_WIDTH_MM

    def estimate_distance_m(
        self,
        real_size_in: float | str | None,
        pixel_size: int,
        frame_width_px: int,
    ) -> float | None:
        """
        Estimate camera-to-object distance in meters.

        real_size_in:   known physical size of the object (inches)
        pixel_size:     size of the same physical extent in pixels
        frame_width_px: width of the frame the pixels were measured in
        """
        try:
            size_in = float(real_size_in)
        except (TypeError, ValueError):
            return None

        if size_in <= 0 or pixel_size <= 0 or frame_width_px <= 0:
            return None

        focal_px = self.focal_length_px(frame_width_px)
        distance_in = size_in * focal_px / pixel_size
        return distance_in / INCHES_PER_METER


def is_fully_visible(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    margin: int = 4,
) -> bool:
    """
    Distance estimates are only reliable when the object is fully in
    frame. A bounding box clipped by a frame edge under-measures the
    object's pixel extent and inflates the distance estimate.
    """
    x1, y1, x2, y2 = box

    return (
        x1 >= margin
        and y1 >= margin
        and x2 <= frame_width - margin
        and y2 <= frame_height - margin
    )


def format_distance(distance_m: float) -> str:
    feet = distance_m * 3.28084
    return f"Distance: {feet:.1f} ft ({distance_m:.1f} m)"


def format_distance_feet(distance_m: float) -> str:
    feet = distance_m * 3.28084
    return f"{feet:.1f} ft"
