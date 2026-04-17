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

import numpy as np

from chemmlapi.core.transforms import INVERSE_TRANSFORMS


def test_expected_keys():
    assert set(INVERSE_TRANSFORMS) == {"none", "log10", "logit"}


def test_none_is_identity():
    x = np.array([-2.5, 0.0, 3.14])
    np.testing.assert_array_equal(INVERSE_TRANSFORMS["none"](x), x)


def test_log10_inverse_roundtrips():
    x = np.array([0.1, 1.0, 100.0])
    np.testing.assert_allclose(INVERSE_TRANSFORMS["log10"](np.log10(x)), x, rtol=1e-12)


def test_logit_inverse_roundtrips():
    p = np.array([0.1, 0.5, 0.9])
    logits = np.log(p / (1 - p))
    np.testing.assert_allclose(INVERSE_TRANSFORMS["logit"](logits), p, rtol=1e-12)
