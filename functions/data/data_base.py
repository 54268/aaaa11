from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataBundle:
    train_known: object
    val_known: object
    test_known: object
    test_unknown: object
    num_known_classes: int
    signal_length: int
    class_names: Optional[list[str]] = None



