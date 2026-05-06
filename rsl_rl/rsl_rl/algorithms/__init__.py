# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms."""

from .ppo import SingleGaitPPO, GaitSelectorPPO

__all__ = ["SingleGaitPPO", "GaitSelectorPPO"]
