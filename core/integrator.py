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
def rk4_step(pos, vel, dt):
    # Standard RK4 Integration for smooth curves
    k1_v = get_acceleration(pos)
    k1_p = vel

    k2_v = get_acceleration(pos + 0.5 * dt * k1_p)
    k2_p = vel + 0.5 * dt * k1_v

    k3_v = get_acceleration(pos + 0.5 * dt * k2_p)
    k3_p = vel + 0.5 * dt * k2_v

    k4_v = get_acceleration(pos + dt * k3_p)
    k4_p = vel + dt * k3_v

    next_pos = pos + (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
    next_vel = vel + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return next_pos, next_vel