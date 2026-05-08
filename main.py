import taichi as ti
import numpy as np
from scene.camera import Camera
import core.constants as c
from core.integrator import rk4_step
import os

ti.init(arch=ti.gpu)

res = (1280, 720)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)
camera = Camera(res[0], res[1])

# --- VIDEO SETTINGS ---
# WICHTIG: automatic_build=False verhindert den Absturz, wenn ffmpeg fehlt
video_manager = ti.tools.VideoManager(output_dir="./render", framerate=30, automatic_build=False)
total_frames = 90


@ti.kernel
def render(t: ti.f32):
    for i, j in pixels:
        u, v = i / res[0], j / res[1]

        # 1. Camera Position (The Orbit)
        radius = 12.0  # Bring camera closer (was 15.0)
        cam_x = ti.cos(t) * radius
        cam_z = ti.sin(t) * radius
        static_cam_pos = ti.Vector([cam_x, 3.0, cam_z])

        # 2. Dynamic Look-At Direction (Ensures the hole stays centered)
        forward = (-static_cam_pos).normalized()
        right = ti.Vector([-ti.sin(t), 0.0, ti.cos(t)]).normalized()
        up = right.cross(forward).normalized()

        screen_dir = camera.get_ray_dir(u, v)
        vel = (screen_dir.x * right + screen_dir.y * up + (-screen_dir.z) * forward).normalized()

        # --- FIX: INITIALIZE COLOR HERE ---
        # This provides a fallback color for every pixel
        color = ti.Vector([0.02, 0.02, 0.05])

        # 3. Ray Integration Loop
        curr_pos = static_cam_pos
        dt = 0.05

        for step in range(1500):
            old_pos = curr_pos
            curr_pos, vel = rk4_step(curr_pos, vel, dt)
            r = curr_pos.norm()

            # --- A. ACCRETION DISK DETECTION ---
            if old_pos.y * curr_pos.y <= 0:
                hit_t = old_pos.y / (old_pos.y - curr_pos.y)
                hit_pos = old_pos + hit_t * (curr_pos - old_pos)
                r_hit = hit_pos.norm()

                if c.DISK_INNER <= r_hit <= c.DISK_OUTER:
                    # Physics: Brightness based on distance and Doppler beaming
                    intensity = 1.0 / (r_hit - c.DISK_INNER + 0.8)
                    gas_vel = ti.Vector([-hit_pos.z, 0.0, hit_pos.x]).normalized()
                    beaming = -vel.dot(gas_vel)
                    doppler_factor = ti.pow(1.0 + beaming * 0.9, 4.0)

                    base_fire = ti.Vector([1.0, 0.4, 0.1])
                    rings = ti.abs(ti.sin(r_hit * 25.0))
                    color = base_fire * (intensity * 2.0) * doppler_factor * (0.4 + 0.6 * rings)
                    break

            # --- B. EVENT HORIZON DETECTION ---
            if r < c.HORIZON_FALLBACK:
                color = ti.Vector([0.0, 0.0, 0.0])
                break

            # --- C. BACKGROUND ESCAPE DETECTION ---
            if r > 100.0:
                direction = vel.normalized()
                phi = ti.atan2(direction.z, direction.x)
                theta = ti.acos(direction.y)
                checker = ti.floor(phi * 3.0) + ti.floor(theta * 6.0)

                if ti.cast(checker, ti.i32) % 2 == 0:
                    color = ti.Vector([0.05, 0.05, 0.1])
                else:
                    color = ti.Vector([0.15, 0.15, 0.25])
                break

        pixels[i, j] = color


#gui = ti.GUI("EventHorizon-RT: Phase 1", res)
#while gui.running:
#    render()
#    gui.set_image(pixels)
#    gui.show()

# --- MAIN RENDER LOOP ---
print("Starte Cinematic Render...")
for frame in range(total_frames):
    t = 2 * np.pi * (frame / total_frames)
    render(t)

    img = pixels.to_numpy()
    video_manager.write_frame(img)
    print(f"Frame {frame + 1}/{total_frames} fertig", end='\r')

print("\nAlle Frames gespeichert unter ./render/")

# Versuche das Video zu bauen, aber fange Fehler ab
try:
    video_manager.make_video(gif=False, mp4=True)
    print("MP4 erfolgreich erstellt.")
except:
    print("\nHinweis: FFmpeg wurde nicht gefunden. Die Einzelbilder wurden in './render' gespeichert.")
    print("Du kannst sie mit einem Programm deiner Wahl zu einem Video zusammenfügen.")