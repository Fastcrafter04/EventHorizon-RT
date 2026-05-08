# core/constants.py
import taichi as ti

M = 1.0
a = 0.95         # Spin parameter (0.0 = Schwarzschild, 0.99 = Ultra-fast)
RS = 2.0 * M
# The event horizon radius changes with spin
HORIZON_FALLBACK = M + ti.sqrt(M**2 - a**2) + 0.01
DISK_INNER = 3.0 * M
DISK_OUTER = 10.0 * M