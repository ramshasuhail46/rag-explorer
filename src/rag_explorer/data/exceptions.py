"""Custom exceptions for the data layer."""


class DataLoadError(Exception):
    """Raised when data cannot be read or no usable rows are found."""

    