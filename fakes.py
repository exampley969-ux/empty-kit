"""
FakeVideoSource: a VideoSource implementation with no real camera/file
dependency, so ingestion logic (connect, reconnect, session lifecycle,
fps tracking) can be tested deterministically and offline.
"""

import numpy as np

from vision_radar.ingestion.video_source import VideoSource


class FakeVideoSource(VideoSource):
    def __init__(self, config, total_frames=10, fail_at_open_times=0,
                 drop_after_n_frames=None, width=64, height=48):
        super().__init__(config)
        self.total_frames = total_frames
        self._frames_emitted = 0
        self._opens_attempted = 0
        self._fail_at_open_times = fail_at_open_times  # how many opens should fail before succeeding
        self._drop_after_n_frames = drop_after_n_frames  # simulate a dropped connection
        self._width = width
        self._height = height
        self._is_open = False

    def _open(self) -> None:
        self._opens_attempted += 1
        if self._opens_attempted <= self._fail_at_open_times:
            raise ConnectionError("simulated open failure")
        self._is_open = True

    def _close(self) -> None:
        self._is_open = False

    def _read_raw(self):
        if not self._is_open:
            return False, None
        if self._drop_after_n_frames is not None and self._frames_emitted == self._drop_after_n_frames:
            # simulate one dropped read; source stays "open" but caller sees
            # the base VideoSource trigger a reconnect
            self._is_open = False
            self._drop_after_n_frames = None  # only drop once
            return False, None
        if self._frames_emitted >= self.total_frames:
            return False, None
        self._frames_emitted += 1
        image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        return True, image

    def native_properties(self) -> dict:
        return {"width": self._width, "height": self._height, "fps": 30.0}

    def is_exhausted(self) -> bool:
        # A pending simulated drop is a live failure, not a clean end.
        if self._drop_after_n_frames is not None:
            return False
        return self._frames_emitted >= self.total_frames
