"""HSN Code Finder - plain-English product description -> likely HSN codes."""
from .matcher import Config, HSNMatcher, Match, strength_label

__all__ = ["Config", "HSNMatcher", "Match", "strength_label"]
__version__ = "1.0.0"
