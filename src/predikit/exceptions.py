class LowConfidenceError(Exception):
    """Raised when a classifier's confidence falls below the configured threshold."""
