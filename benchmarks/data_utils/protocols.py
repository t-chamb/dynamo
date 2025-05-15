"""
Protocol-level constants for synthetic data graph structure.
"""

SUPER_ROOT = -1  # Dummy node preceding all real nodes; not an actual data root
CACHE_END = -2  # Special node indicating end of a path
END_NODE = -3  # Special node indicating to skip leaf sampling
