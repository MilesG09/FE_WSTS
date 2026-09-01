from .BaseModel import BaseModel
from .SMPModel import SMPModel
from .SMPTempModel import SMPTempModel
from .ConvLSTMLightning import ConvLSTMLightning
from .LogisticRegression import LogisticRegression

# Optional models pull in heavier / not-always-installed deps (e.g. transformers)
# or files that were removed from this fork. Guard them so a UNet/UTAE run does
# not require the full stack.
try:
    from .UTAELightning import UTAELightning
    from .UTAELightningDumb import UTAELightningDumb
except Exception as _e:  # noqa: BLE001
    print(f"[models/__init__] skipping UTAE models: {_e}")
try:
    from .SwinUnetLightning import SwinUnetLightning
    from .SwinUnetTempLightning import SwinUnetTempLightning
except Exception as _e:  # noqa: BLE001
    print(f"[models/__init__] skipping SwinUnet models: {_e}")
try:
    from .SegFormerLightning import SegFormerLightning
except Exception as _e:  # noqa: BLE001
    print(f"[models/__init__] skipping SegFormer model: {_e}")
