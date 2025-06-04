from enum import IntEnum


class Finger(IntEnum):
    THUMB = 4
    INDEX = 8
    MIDDLE = 12
    RING = 16
    PINKY = 20


def is_index_finger_only_up_mediapipe(lm) -> bool:
    index_up = lm[Finger.INDEX].y < lm[Finger.INDEX - 2].y
    middle_dn = lm[Finger.MIDDLE].y > lm[Finger.MIDDLE - 2].y
    ring_dn = lm[Finger.RING].y > lm[Finger.RING - 2].y
    pinky_dn = lm[Finger.PINKY].y > lm[Finger.PINKY - 2].y
    thumb_in = lm[Finger.THUMB].x > lm[Finger.THUMB - 1].x
    return index_up and middle_dn and ring_dn and pinky_dn and thumb_in


def is_open_palm(lm) -> bool:
    for tip in (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY):
        if not lm[tip].y < lm[tip - 2].y:
            return False
    return True


def is_index_finger_only_up_model(kpts) -> bool:
    """Check if only index finger is up using model output keypoints [21, 2]"""
    def y(k): return kpts[k][1]  # y-coordinate

    index_up = y(8) < y(6)
    middle_dn = y(12) > y(10)
    ring_dn = y(16) > y(14)
    pinky_dn = y(20) > y(18)
    thumb_in = kpts[4][0] > kpts[3][0]

    return index_up and middle_dn and ring_dn and pinky_dn and thumb_in
