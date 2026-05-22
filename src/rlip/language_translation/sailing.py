"""
Sailing Language Translator
=============================
Maps the ``"x_angle"`` observation string from :class:`~rlip.environments.predefined.sailing.SailingEnvironment`
to a natural-language description of the sailboat's position, heading, and
last action.

State format
------------
The observation is a string ``"{x:.4f}_{angle:.1f}"``, e.g. ``"0.0300_0.2"``.

    x     - horizontal position  (negative = harbor side, positive = beach side)
    angle - heading in radians   (0 = directly into wind, ±π/2 = across wind)

Actions
-------
    0 - turn slightly left  (angle -= 0.1 rad)
    1 - turn slightly right (angle += 0.1 rad)

Example output
--------------
    "The boat is in the middle, close hauled with wind on the port side,
     the last action was to turn towards the beach."
"""

from __future__ import annotations

from typing import Any

from .base import LanguageTranslator


class SailingLanguageTranslator(LanguageTranslator):
    """Language translator for ``Sailing-v0`` and ``Sailing-Hard-v0``."""

    name = "sailing"

    def translate(
        self,
        state: Any,
        *,
        legal_moves: list[Any] | None = None,
        action_history: list[Any] | None = None,
    ) -> str:
        """
        Convert a ``"x_angle"`` sailing observation into plain English.

        Parameters
        ----------
        state:
            Observation string, e.g. ``"0.0300_0.2"``.
        legal_moves:
            Not used by this translator (both actions are always legal).
        action_history:
            List of past actions (integers 0 or 1).  The last entry is used
            to describe what move was just taken.

        Returns
        -------
        str
            A readable description of the current sailing state.
        """
        try:
            parts = str(state).split("_")
            x = float(parts[0])
            angle = float(parts[1])
        except (IndexError, ValueError):
            return f"Unknown sailing state: {state!r}"

        # ── Horizontal position description ───────────────────────────────────
        if -1 < x < 1:
            L_x = "in the middle"
        elif -3 < x < 3:
            L_x = "near to the center"
        elif -5 < x < 5:
            L_x = "in between the edge and the center"
        elif -7 < x < 7:
            L_x = "near to the edge"
        elif -10 <= x <= 10:
            L_x = "very close to the edge"
        else:
            L_x = "out of bounds"

        # ── Side of river ─────────────────────────────────────────────────────
        if x < 0:
            L_x_side = "on the harbor side of the river"
        elif x > 0:
            L_x_side = "on the beach side of the river"
        else:
            L_x_side = ""

        # ── Heading / angle description ───────────────────────────────────────
        if angle == 0:
            L_angle = "facing directly into the wind"
        elif -0.1 < angle < 0.1:
            L_angle = "facing into the wind"
        elif -0.5 < angle < 0.5:
            L_angle = "close hauled with wind"
        elif -1 < angle < 1:
            L_angle = "cutting the wind"
        else:
            L_angle = "moving across the wind"

        # ── Wind side ─────────────────────────────────────────────────────────
        if angle < 0:
            L_wind_side = "on the starboard side"
        elif angle > 0:
            L_wind_side = "on the port side"
        else:
            L_wind_side = ""

        # ── Combine position and heading ──────────────────────────────────────
        parts_text = [p for p in [L_x_side, L_x] if p]
        L_state = "The boat is " + ", ".join(parts_text) + ", " + L_angle
        if L_wind_side:
            L_state += " " + L_wind_side
        L_state += ","

        # ── Last action ───────────────────────────────────────────────────────
        L_action = ""
        if action_history:
            last_action = action_history[-1]
            if x <= 0 and last_action == 0:
                L_action = "the last action was to turn towards the harbor."
            elif x < 0 and last_action == 1:
                L_action = "the last action was to turn towards the center of the river."
            elif x >= 0 and last_action == 1:
                L_action = "the last action was to turn towards the beach."
            elif x > 0 and last_action == 0:
                L_action = "the last action was to turn towards the center of the river."

        # ── Clean up and return ───────────────────────────────────────────────
        result = (L_state + " " + L_action).strip() if L_action else L_state.rstrip(",") + "."
        result = (
            result
            .replace("  ", " ")
            .replace(" .", ".")
            .replace(" ,", ",")
        )
        return result
