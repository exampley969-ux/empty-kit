"""
Vision Radar - World Model: Zones
Zone definition (polygon/rectangle) and live occupancy.

Spec reference: PDF Section 8 (Spatial model - Zones: "Users can draw
polygon/rectangular zones"), Section 6 (World Model example -
"Zone A: Occupancy: 2, Entities: #17,#04, State: Active").
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

from vision_radar.world_model.models import EntitySnapshot


class ZoneState(str, Enum):
    """Matches the Section 6 example field ("State: Active")."""
    ACTIVE = "ACTIVE"   # at least one entity currently inside
    IDLE = "IDLE"       # zone defined, currently empty


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Standard ray-casting point-in-polygon test. Works for both an
    arbitrary polygon and the 4-corner rectangle special case.
    A point exactly on the boundary may be classified either way (a known,
    accepted ambiguity of this algorithm) - not a concern for entity
    occupancy at real-world frame rates/resolutions.
    """
    x, y = point
    n = len(polygon)
    inside = False
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if y > min(y1, y2):
            if y <= max(y1, y2):
                if x <= max(x1, x2):
                    if y1 != y2:
                        x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    if x1 == x2 or x <= x_intersect:
                        inside = not inside
        x1, y1 = x2, y2
    return inside


@dataclass
class Zone:
    """A user-defined monitoring region, in image-space pixel coordinates
    (Section 15 - no real-world units claimed without real calibration)."""
    zone_id: str
    name: str
    polygon: List[Tuple[float, float]]   # >= 3 points, in pixel coordinates

    @classmethod
    def from_rect(cls, name: str, x: float, y: float, width: float, height: float,
                  zone_id: str = None) -> "Zone":
        """Convenience constructor for the common rectangular-zone case."""
        polygon = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        return cls(zone_id=zone_id or str(uuid.uuid4()), name=name, polygon=polygon)

    @classmethod
    def from_polygon(cls, name: str, points: List[Tuple[float, float]],
                      zone_id: str = None) -> "Zone":
        if len(points) < 3:
            raise ValueError("A zone polygon needs at least 3 points")
        return cls(zone_id=zone_id or str(uuid.uuid4()), name=name, polygon=list(points))

    def contains(self, point: Tuple[float, float]) -> bool:
        return point_in_polygon(point, self.polygon)


@dataclass
class ZoneOccupancy:
    """Live per-zone snapshot, shaped directly like the Section 6 example."""
    zone_id: str
    zone_name: str
    entity_ids: List[str] = field(default_factory=list)
    state: ZoneState = ZoneState.IDLE

    @property
    def occupancy(self) -> int:
        return len(self.entity_ids)


class ZoneManager:
    """Owns the set of defined zones and computes live occupancy against
    the World Model's current entities, once per frame."""

    def __init__(self):
        self._zones: Dict[str, Zone] = {}

    def add_zone(self, zone: Zone) -> None:
        self._zones[zone.zone_id] = zone

    def remove_zone(self, zone_id: str) -> None:
        del self._zones[zone_id]

    def get_zone(self, zone_id: str) -> Zone:
        return self._zones[zone_id]

    def list_zones(self) -> List[Zone]:
        return list(self._zones.values())

    def compute_occupancy(self, entities: List[EntitySnapshot]) -> Dict[str, ZoneOccupancy]:
        """
        Returns one ZoneOccupancy per defined zone (even zones with zero
        entities are included, as IDLE) - callers should not need to
        special-case "zone not mentioned = empty".

        Note: a single entity can appear in more than one zone if zones
        overlap - this is intentional (Section 8 imposes no
        non-overlap constraint), and each zone's occupancy is computed
        independently.
        """
        result: Dict[str, ZoneOccupancy] = {
            zid: ZoneOccupancy(zone_id=zid, zone_name=zone.name)
            for zid, zone in self._zones.items()
        }

        for entity in entities:
            for zid, zone in self._zones.items():
                if zone.contains(entity.position):
                    result[zid].entity_ids.append(entity.entity_id)

        for occ in result.values():
            occ.state = ZoneState.ACTIVE if occ.entity_ids else ZoneState.IDLE

        return result
