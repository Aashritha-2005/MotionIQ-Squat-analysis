"""Task 4.1: causal, streaming rep counter over per-frame phase predictions (argmax of the ONNX phase model).

The model's raw output rarely flickers (~4.2 phase changes per reference rep on val subjects; a clean rep is 4),
so smoothing is applied where it matters for counting -- the standing <-> moving boundary -- instead of a
per-phase dwell (which swallowed genuinely short bottom phases and under-counted on the validation subjects):

  * hysteresis: "moving" starts after `k_in` consecutive non-standing frames, ends after `k_out` consecutive
    standing frames (shorter runs are ignored as flicker)
  * bounce boundary: a rep also ends when the prediction goes ascent -> descent and the descent lasts `k_bounce`
    frames (continuous reps without fully standing up); k_bounce = 0 disables this
  * a closed segment counts as a rep if its non-standing raw phases contain the `require`d phases in order
    ("dba" = descent, bottom, ascent; "da" = descent ... ascent; "ba"; "a"). A clip that starts mid-squat
    (no descent before the first ascent) is not counted under "dba"/"da".
Parameters are tuned on the internal-validation subjects (scripts/evaluate_phase4.py), never on held-out.

Usage: rc = RepCounter(**cfg); ev = rc.update(t, phase)  -> None or {"rep": n, "start": frame, "end": frame}
       (decided only from frames <= t; `start`/`end` are back-dated to the segment boundaries)
"""
from collections import deque

PHASES = ["standing", "descent", "bottom", "ascent"]
STANDING, DESCENT, BOTTOM, ASCENT = range(4)
REQUIRE = {"dba": (DESCENT, BOTTOM, ASCENT), "da": (DESCENT, ASCENT), "ba": (BOTTOM, ASCENT), "a": (ASCENT,)}


class RepCounter:
    def __init__(self, k_in=2, k_out=1, k_bounce=2, require="da"):
        self.k_in, self.k_out, self.k_b, self.req = int(k_in), int(k_out), int(k_bounce), REQUIRE[require]
        self.moving, self.run, self.count = False, 0, 0
        self.recent = deque(maxlen=max(self.k_in, 1))
        self.seq, self.start = [], None    # raw phases of the open segment and its first frame
        self.phase = STANDING               # last raw phase (for display)

    @property
    def state(self):
        """Displayed phase: raw phase while moving, standing otherwise."""
        return self.phase if self.moving else STANDING

    def _close(self, seq, start, end):
        ex = [p for p in seq if p != STANDING]
        it = iter(ex)
        if all(r in it for r in self.req):
            self.count += 1
            return {"rep": self.count, "start": start, "end": end}
        return None

    def reject_last(self):
        """Un-count the rep just returned by update() (used by the pipeline's framing gate)."""
        self.count -= 1

    def update(self, t, phase, prob=None):
        ph = int(phase)
        self.phase = ph
        self.recent.append(ph)
        ev = None
        if self.moving:
            self.seq.append(ph)
            kb = self.k_b
            if kb and len(self.seq) > kb and self.seq[-kb - 1] == ASCENT and all(p == DESCENT for p in self.seq[-kb:]):
                ev = self._close(self.seq[:-kb], self.start, t - kb)
                self.seq, self.start = self.seq[-kb:], t - kb + 1
        self.run = self.run + 1 if (ph != STANDING) != self.moving else 0
        if self.run >= (self.k_out if self.moving else self.k_in):
            self.moving, self.run = not self.moving, 0
            if self.moving:
                self.seq, self.start = list(self.recent)[-self.k_in:], t - self.k_in + 1
            else:
                ev = self._close(self.seq, self.start, t - self.k_out) or ev
                self.seq, self.start = [], None
        return ev
