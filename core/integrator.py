# core/integrator.py
import taichi as ti
import core.constants as c


@ti.func
def get_acceleration(pos):
    # Newton was almost right, but Einstein adds a 'kick'
    # This function defines how gravity bends the light path
    r2 = pos.dot(pos)
    r = ti.sqrt(r2)
    # Schwarzschild acceleration approximation for photons
    accel = -1.5 * c.RS * pos / (r2 * r2)
    return accel


@ti.func
def get_derivatives(pos, p):
    # Boyer-Lindquist coordinates/logic simplified for Cartesian integration
    x, y, z = pos.x, pos.y, pos.z
    r2 = pos.dot(pos)
    r = ti.sqrt(r2)
    a2 = c.a ** 2

    # Sigma and Delta terms essential for Kerr geometry
    sigma = r2 + a2 * (z ** 2 / r2)
    delta = r2 - 2.0 * c.M * r + a2

    # This is a simplified "effective potential" approach for rays
    # For true Kerr, we calculate dp/dt based on the metric tensor
    force_magnitude = (c.M / (sigma ** 2)) * (r2 - a2)

    # Frame-dragging: The spin 'a' creates a lateral kick
    drag = 2.0 * c.M * r * c.a / (sigma * r2)

    accel = -force_magnitude * pos
    accel.x += drag * p.y  # Spacetime dragging the ray around the Z-axis
    accel.y -= drag * p.x

    return p, accel


@ti.func
def rk4_step(pos, p, dt):
    # RK4 remains the same, but now operates on (Position, Momentum) pairs
    k1_p, k1_a = get_derivatives(pos, p)

    k2_p, k2_a = get_derivatives(pos + 0.5 * dt * k1_p, p + 0.5 * dt * k1_a)
    k3_p, k3_a = get_derivatives(pos + 0.5 * dt * k2_p, p + 0.5 * dt * k2_a)
    k4_p, k4_a = get_derivatives(pos + dt * k3_p, p + dt * k3_a)

    next_pos = pos + (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
    next_p = p + (dt / 6.0) * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)

    return next_pos, next_p