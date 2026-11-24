"""The observation contract's own values for a point name and a dimension.

Standard library only. These values are defined here and are not read from ``nvs_edge``.
When ``nvs_edge`` is installed they are expected to equal its contract constants, which the
integration tests check.
"""

from __future__ import annotations

import re
from typing import Final

#: The largest hidden-state dimension a capture may record.
MAX_DIMENSION: Final = 65535

#: Point names: ``final_norm``, or ``block_out`` / ``mlp_out`` and a decimal index
#: (``0``, or a number with no leading zero).
POINT_PATTERN: Final = re.compile(r"final_norm|(?:block_out|mlp_out):(?:0|[1-9][0-9]*)")
