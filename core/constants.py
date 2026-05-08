import taichi as ti

# Unit Consistency: G = c = 1
# This makes the Schwarzschild radius R = 2 * M
M = 1.0          # Mass of the black hole
RS = 2.0 * M     # Schwarzschild Radius
HORIZON_FALLBACK = 1.01 * RS # Stop rays slightly before they hit to avoid math errors
DISK_INNER = 3.0 * M  # Innermost Stable Circular Orbit (ISCO)
DISK_OUTER = 10.0 * M