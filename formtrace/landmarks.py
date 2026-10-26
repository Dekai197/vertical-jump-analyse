"""BlazePose (MediaPipe Pose Landmarker) 33-point indices and helpers."""

from __future__ import annotations

import numpy as np

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32  # "foot index" = toe tip

SIDES = {
    "left": dict(shoulder=L_SHOULDER, elbow=L_ELBOW, wrist=L_WRIST, hip=L_HIP,
                 knee=L_KNEE, ankle=L_ANKLE, heel=L_HEEL, toe=L_FOOT),
    "right": dict(shoulder=R_SHOULDER, elbow=R_ELBOW, wrist=R_WRIST, hip=R_HIP,
                  knee=R_KNEE, ankle=R_ANKLE, heel=R_HEEL, toe=R_FOOT),
}

# Bones drawn on the overlay (body only; face points are noisy and irrelevant here).
BONES = [
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (L_ANKLE, L_HEEL), (L_HEEL, L_FOOT), (L_ANKLE, L_FOOT),
    (R_ANKLE, R_HEEL), (R_HEEL, R_FOOT), (R_ANKLE, R_FOOT),
]


def camera_side(vis: np.ndarray) -> str:
    """Pick the body side facing the camera: higher mean visibility of hip/knee/ankle.

    ``vis`` has shape (T, 33). In a true side view the far-side joints are
    occluded and MediaPipe *guesses* them, so all single-side metrics use the
    near side.
    """
    def score(side: str) -> float:
        s = SIDES[side]
        return float(np.nanmean(vis[:, [s["hip"], s["knee"], s["ankle"]]]))
    return "left" if score("left") >= score("right") else "right"
