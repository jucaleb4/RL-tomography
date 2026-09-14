from enum import Enum

class ImageType(Enum):
    MIXED = 0
    TRIANGLE = 1
    TRIANGLE_THIRTY = 2
    TRIANGLE_FIXED = 3

class RewardType(Enum):
    PNSR_E2E = 0
    PNSR_INCREMENTAL = 1
    FWD_E2E = 2
    FWD_INCREMENTAL = 3
