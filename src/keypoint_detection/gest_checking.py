from enum import IntEnum


class Finger(IntEnum):
    """Indices of fingertip landmarks in MediaPipe Hands."""

    THUMB = 4
    INDEX = 8
    MIDDLE = 12
    RING = 16
    PINKY = 20


def is_index_finger_only_up(lm) -> bool:
    """Return True when only the index finger is raised"""
    index_up = lm[Finger.INDEX].y < lm[Finger.INDEX - 2].y  # tip above PIP
    middle_dn = lm[Finger.MIDDLE].y > lm[Finger.MIDDLE - 2].y
    ring_dn = lm[Finger.RING].y > lm[Finger.RING - 2].y
    pinky_dn = lm[Finger.PINKY].y > lm[Finger.PINKY - 2].y
    thumb_in = lm[Finger.THUMB].x > lm[Finger.THUMB - 1].x
    return index_up and middle_dn and ring_dn and pinky_dn and thumb_in


def is_open_palm(lm) -> bool:
    """Return True if all five fingers appear extended/open."""
    for tip in (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY):
        if not lm[tip].y < lm[tip - 2].y:
            return False
    return True
