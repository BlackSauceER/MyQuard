# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .dreamwaq import DreamWaQ
from .gait_selector import GaitSelector

__all__ = [
    "DreamWaQ",
    "GaitSelector"
]
