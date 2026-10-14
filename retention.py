"""
Vision Radar - Data Layer: retention
Configurable retention, so the system never quietly keeps everything.

Spec reference: PDF Section 28 (Data protection: "no unnecessary raw-video
persistence, configurable retention"), Section 29 (Privacy: "configurable
retention, structured-event storage without automatically retaining
everything"), Section 32 ("retention behaviour is defined").

Two clocks, because they hold different kinds of data:
  * evidence IMAGES (frames that may show people) expire fast - default 7 days.
    Only the image file is removed; the structured evidence row stays until
    the event itself expires.
  * finished EVENTS (and, by cascade, their alerts/evidence rows) expire
    after `event_retention_days` - default 30.
The defaults are placeholders: the spec requires retention to be
*configurable and defined*, not any particular number.

Timestamps compared here must be epoch seconds (the pipeline runner stamps
events with real capture time). ACTIVE events are never deleted.

Safety: an image path is only ever deleted if it resolves to somewhere
inside the configured evidence directory, so a tampered database row can't
turn retention into "delete any file on disk".
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from .repository import Repository

_DAY = 86400.0


@dataclass
class RetentionPolicy:
    event_retention_days: float = 30.0
    evidence_image_retention_days: float = 7.0

    def validate(self) -> None:
        if self.event_retention_days <= 0 or self.evidence_image_retention_days <= 0:
            raise ValueError("Retention periods must be greater than zero days")


@dataclass
class RetentionReport:
    events_deleted: int = 0
    evidence_images_deleted: int = 0
    images_skipped_outside_evidence_dir: int = 0


def _delete_if_inside(path: str, evidence_dir: str) -> Optional[bool]:
    """True = deleted, False = file already gone, None = refused (outside dir)."""
    root = os.path.realpath(evidence_dir)
    target = os.path.realpath(path)
    if not target.startswith(root + os.sep):
        return None
    try:
        os.remove(target)
        return True
    except FileNotFoundError:
        return False


def apply_retention(repo: Repository, policy: RetentionPolicy, evidence_dir: str,
                    now: Optional[float] = None) -> RetentionReport:
    policy.validate()
    now = time.time() if now is None else now
    report = RetentionReport()

    # 1. old evidence images
    image_cutoff = now - policy.evidence_image_retention_days * _DAY
    for event_id, path in repo.evidence_with_files_older_than(image_cutoff):
        outcome = _delete_if_inside(path, evidence_dir)
        if outcome is None:
            report.images_skipped_outside_evidence_dir += 1
            continue
        if outcome:
            report.evidence_images_deleted += 1
        repo.clear_evidence_file_path(event_id)

    # 2. old finished events (their images go with them)
    event_cutoff = now - policy.event_retention_days * _DAY
    ids = repo.finished_event_ids_older_than(event_cutoff)
    if ids:
        for path in repo.evidence_file_paths_for(ids):
            outcome = _delete_if_inside(path, evidence_dir)
            if outcome is True:
                report.evidence_images_deleted += 1
            elif outcome is None:
                report.images_skipped_outside_evidence_dir += 1
        report.events_deleted = repo.delete_events(ids)

    return report
