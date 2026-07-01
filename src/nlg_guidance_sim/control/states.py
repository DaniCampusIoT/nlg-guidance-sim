from enum import Enum


class CaptureState(str, Enum):
    ACQUISITION = "acquisition"
    APPROACH = "approach"
    CENTERING_CAPTURE = "centering_capture"
    LIFT_PERMISSION = "lift_permission"
