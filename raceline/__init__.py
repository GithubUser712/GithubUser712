"""RC raceline pipeline — map ingestion through hardware deployment.

Subpackages
-----------
core       Shared types, validation, artifact I/O
maps       Track image loading, centerline extraction
physics    High-fidelity vehicle dynamics (tires, motor, integrator)
sim        2D Gymnasium environment and lidar
control    Pure pursuit and trajectory tracking
hardware   VESC UART driver
viewer     Training / validation visualization
gazebo     Gazebo world export and ROS 2 helpers
"""

__version__ = "2.0.0"
