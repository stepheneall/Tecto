"""
Pinhole-camera binocular renderer.

Renders what the fish sees through its two eyes. Each eye is a 280×280 pixel
image with a pinhole camera model (focal length fl=80, 100° FOV).

Eyes are offset ±3.0 units on X (horizontal) and angled ±25° outward.
There is zero Y offset — meaning vertical disparity is unavailable.
"""
import numpy as np
import cv2
import math

AX = 36; AY0, AY1 = 0, 30; AZ = 60
EYE_OFFSET = 3.0                     # Horizontal eye separation
EYE_ANGLE = math.radians(25.0)       # Eye angle outward from heading
IMAGE_SIZE = (280, 280)              # (height, width) in pixels
FOCAL_LENGTH = 80                    # Pinhole camera focal length


def _rot(dx, dz, angle):
    """Rotate a 2D point by `angle` radians."""
    return (dx * math.cos(angle) - dz * math.sin(angle),
            dx * math.sin(angle) + dz * math.cos(angle))


def render(fx, fy, fz, fh, foods, obstacles):
    """
    Render stereo pair from the fish's perspective.

    Args:
        fx, fy, fz: fish position (world coordinates).
        fh:         fish heading angle (radians, 0 = forward along +Z).
        foods:      list of [x, y, z, radius] for each food item.
        obstacles:  list of [x, y, z, rx, ry, rz] for each obstacle.

    Returns:
        (left_img, right_img): two 280×280×3 uint8 numpy arrays (BGR).
    """
    sz = IMAGE_SIZE
    cx, cy = sz[1] // 2, sz[0] // 2
    fl = FOCAL_LENGTH

    # Eye positions and orientations
    L_ang = fh - EYE_ANGLE
    R_ang = fh + EYE_ANGLE
    L_ex = fx - EYE_OFFSET * math.cos(fh)
    L_ez = fz + EYE_OFFSET * math.sin(fh)
    R_ex = fx + EYE_OFFSET * math.cos(fh)
    R_ez = fz - EYE_OFFSET * math.sin(fh)

    imgs = {}
    for ex, ez, eh, lbl in [(L_ex, L_ez, L_ang, 'L'),
                              (R_ex, R_ez, R_ang, 'R')]:
        img = np.zeros((*sz, 3), np.uint8)

        # --- Background: simple gradient (simulates water haze) ---
        for py in range(sz[0]):
            v = int(60 + 80 * py / sz[0])
            img[py, :] = [v, int(v * 0.7), 40]

        # --- Obstacles (gray boxes) ---
        for ox, oy, oz, orx, ory, orz in obstacles:
            # Transform to eye coordinates
            rlx, rlz = _rot(ox - ex, oz - ez, eh)
            rly = oy - fy
            d = math.sqrt(rlx**2 + rly**2 + rlz**2)
            if d >= 0.5 and rlz > 0.3:  # In front of the eye
                # Project to image plane
                px = int(cx + fl * rlx / max(rlz, 0.5))
                py = int(cy + fl * rly / max(rlz, 0.5))
                prx = max(2, int(fl * orx / max(rlz, 0.5)))
                pry = max(2, int(fl * ory / max(rlz, 0.5)))
                cv2.rectangle(img,
                    (max(0, px - prx), max(0, py - pry)),
                    (min(sz[1] - 1, px + prx), min(sz[0] - 1, py + pry)),
                    (80, 80, 80), -1)

        # --- Food (green spheres) ---
        for ffx, ffy, ffz, fr in foods:
            rlx, rlz = _rot(ffx - ex, ffz - ez, eh)
            rly = ffy - fy
            d = math.sqrt(rlx**2 + rly**2 + rlz**2)
            if d >= 0.5 and rlz > 0.3:
                px = int(cx + fl * rlx / max(rlz, 0.5))
                py = int(cy + fl * rly / max(rlz, 0.5))
                pr = max(3, int(fl * fr / max(rlz, 0.5)))
                if 0 <= px < sz[1] and 0 < py < sz[0]:
                    cv2.circle(img, (px, py), pr, (0, 255, 0), -1)

        imgs[lbl] = img

    return imgs['L'], imgs['R']
