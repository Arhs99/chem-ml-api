# chem-ml-api - FastAPI inference service for chemprop ADMET models
# Copyright (C) 2026  Kostas Papadopoulos <kostasp97@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import unittest
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
LOGD_CHECKPOINT_DIR = FIXTURES_DIR / "logd"


def _has_logd_checkpoint() -> bool:
    if not LOGD_CHECKPOINT_DIR.is_dir():
        return False
    if (LOGD_CHECKPOINT_DIR / "best.pt").is_file():
        return True
    if any(LOGD_CHECKPOINT_DIR.glob("*/best.pt")):
        return True
    if any(LOGD_CHECKPOINT_DIR.glob("*/checkpoints/*.ckpt")):
        return True
    return False


REQUIRES_LOGD = unittest.skipUnless(
    _has_logd_checkpoint(),
    f"logD checkpoint not present at {LOGD_CHECKPOINT_DIR}",
)
