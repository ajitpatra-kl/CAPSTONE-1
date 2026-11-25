"""Persistent LCVD configuration and validated detection settings."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import isfinite
import json

from modules.rlvd.configuration import save_configuration

from .lane_geometry import LaneGeometry


@dataclass(frozen=True)
class Settings:
    stable_frames: int = 5
    margin: float = 4.0
    confidence: float = .35
    max_gap: int = 12
    max_jump: float = .70

    def validate(self):
        if not (isinstance(self.stable_frames, int) and 2 <= self.stable_frames <= 30
                and 0 <= self.margin <= 40 and .1 <= self.confidence <= .99
                and isinstance(self.max_gap, int) and 1 <= self.max_gap <= 60
                and .15 <= self.max_jump <= 1):
            raise ValueError("Lane confirmation settings are outside the supported range.")


@dataclass
class Configuration:
    name: str
    video_id: str
    width: int
    height: int
    fps: float
    lines: list
    axis: str = "Vertical"
    reverse: bool = False
    settings: Settings = field(default_factory=Settings)
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def geometry(self):
        if not self.name.strip() or self.width < 1 or self.height < 1 or not isfinite(self.fps) or self.fps <= 0:
            raise ValueError("A name and valid video metadata are required.")
        self.settings.validate()
        return LaneGeometry(self.width, self.height, self.lines, self.axis, self.reverse)

    def save(self, folder, config_id):
        self.geometry()
        save_configuration(folder, asdict(self), config_id)

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        data["settings"] = Settings(**data.get("settings", {}))
        config = cls(**data)
        config.geometry()
        return config
