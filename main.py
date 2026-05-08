import taichi as ti
import numpy as np
from scene.camera import Camera
import core.constants as c
from core.integrator import rk4_step
import os

ti.init(arch=ti.gpu)

res = (1280, 720)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)
# Neues Feld für den Bloom-Effekt
bloom_pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)
camera = Camera(res[0], res[1])

video_manager = ti.tools.VideoManager(output_dir="./render", framerate=30, automatic_build=False)
total_frames = 90


@ti.kernel
def render(t: ti.f32):
    for i, j in pixels:
        u, v = i / res[0], j / res[1]
        radius = 14.0
        cam_x = ti.cos(t) * radius
        cam_z = ti.sin(t) * radius
        static_cam_pos = ti.Vector([cam_x, 3.5, cam_z])

        forward = (-static_cam_pos).normalized()
        right = ti.Vector([-ti.sin(t), 0.0, ti.cos(t)]).normalized()
        up = right.cross(forward).normalized()

        screen_dir = camera.get_ray_dir(u, v)
        vel = (screen_dir.x * right + screen_dir.y * up + (-screen_dir.z) * forward).normalized()

        color = ti.Vector([0.005, 0.005, 0.01])
        curr_pos = static_cam_pos
        dt = 0.06

        for step in range(1600):
            old_pos = curr_pos
            curr_pos, vel = rk4_step(curr_pos, vel, dt)
            r = curr_pos.norm()

            if ti.abs(curr_pos.y) < 0.15:
                r_hit = ti.sqrt(curr_pos.x ** 2 + curr_pos.z ** 2)
                if c.DISK_INNER <= r_hit <= c.DISK_OUTER:
                    intensity = 1.0 / (r_hit - c.DISK_INNER + 0.6)
                    gas_vel = ti.Vector([-curr_pos.z, 0.0, curr_pos.x]).normalized()
                    beaming = -vel.dot(gas_vel)
                    doppler = ti.pow(1.0 + beaming * 0.85, 4.0)
                    noise = ti.abs(ti.sin(r_hit * 20.0) * ti.cos(ti.atan2(curr_pos.z, curr_pos.x) * 10.0))

                    glow = ti.Vector([1.0, 0.45, 0.15]) * intensity * doppler * (0.4 + 0.6 * noise)
                    color += glow * 0.2
                    if color.x > 1.5: break

            if r < c.HORIZON_FALLBACK:
                color = ti.Vector([0.0, 0.0, 0.0])
                break

            if r > 120.0:
                direction = vel.normalized()
                star_seed = ti.sin(direction.x * 150.0) * ti.sin(direction.y * 150.0) * ti.sin(direction.z * 150.0)
                if star_seed > 0.992:
                    color += ti.Vector([0.9, 0.9, 1.0]) * star_seed
                break

        pixels[i, j] = color


@ti.kernel
def apply_bloom():
    # Einfacher Box-Blur als Bloom-Ersatz (schnell und effektiv)
    blur_radius = 3
    for i, j in pixels:
        bloom_val = ti.Vector([0.0, 0.0, 0.0])
        count = 0
        for di in range(-blur_radius, blur_radius + 1):
            for dj in range(-blur_radius, blur_radius + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < res[0] and 0 <= nj < res[1]:
                    # Nur Pixel extrahieren, die hell genug zum "Leuchten" sind
                    pix = pixels[ni, nj]
                    brightness = (pix.x + pix.y + pix.z) / 3.0
                    if brightness > 0.5:
                        bloom_val += pix
                    count += 1

        # Originalbild + weichgezeichnetes Glühen
        pixels[i, j] += (bloom_val / count) * 0.6


# --- MAIN RENDER LOOP ---
print("Starte Cinematic Render mit Bloom...")
for frame in range(total_frames):
    t = 2 * np.pi * (frame / total_frames)
    render(t)
    apply_bloom()  # Bloom nach jedem Frame anwenden

    img = pixels.to_numpy()
    video_manager.write_frame(img)
    print(f"Frame {frame + 1}/{total_frames} fertig (Bloom inkl.)", end='\r')

print("\nAlle Frames gespeichert. Baue Video...")
try:
    video_manager.make_video(gif=False, mp4=True)
    print("Video erfolgreich mit Bloom-Effekt erstellt!")
except:
    print("\nFFmpeg Fehler, aber Bilder sind in ./render/")