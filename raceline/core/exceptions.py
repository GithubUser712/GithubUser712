"""Typed errors for every pipeline stage — callers get actionable messages."""


class PipelineError(Exception):
    """Base class for all raceline pipeline failures."""


class ArtifactError(PipelineError):
    """Missing, corrupt, or schema-invalid artifact files."""


class MapValidationError(PipelineError):
    """Track map failed topological or geometric validation."""


class PhysicsError(PipelineError):
    """Vehicle parameter or dynamics integration failure."""


class PreflightError(PipelineError):
    """A pipeline step was started without its prerequisites."""
