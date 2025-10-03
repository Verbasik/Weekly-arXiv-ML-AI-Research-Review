"""
MoE (Mixture-of-Experts) domain module for Qwen3 architecture.

This module contains:
- MoERouter: Routes tokens to top-K experts
- Expert: Individual expert network (feed-forward)
- MoELayer: Complete MoE layer combining router and experts
"""

from .router import MoERouter

__all__ = ['MoERouter']
