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

        # 1. Kamera-Position (Optimiert für Gargantua-Look)
        radius = 14.0  # Näher dran für mehr Details
        cam_x = ti.cos(t) * radius
        cam_z = ti.sin(t) * radius
        # Höherer Y-Wert (3.5) zeigt die Lichtbeugung über dem Loch besser
        static_cam_pos = ti.Vector([cam_x, 3.5, cam_z])

        # 2. Look-At Logic (Kamera fixiert das Zentrum)
        forward = (-static_cam_pos).normalized()
        right = ti.Vector([-ti.sin(t), 0.0, ti.cos(t)]).normalized()
        up = right.cross(forward).normalized()

        screen_dir = camera.get_ray_dir(u, v)
        vel = (screen_dir.x * right + screen_dir.y * up + (-screen_dir.z) * forward).normalized()

        # Initialisierung: Fast schwarz mit leichtem blauen Schimmer für die Tiefe
        color = ti.Vector([0.005, 0.005, 0.01])
        curr_pos = static_cam_pos
        dt = 0.06  # Balance zwischen Speed und Präzision

        for step in range(1600):
            old_pos = curr_pos
            curr_pos, vel = rk4_step(curr_pos, vel, dt)
            r = curr_pos.norm()

            # --- A. VERBESSERTE AKKRETIONSSCHEIBE (INTERSTELLAR STYLE) ---
            # Wir geben der Scheibe eine kleine Dicke (0.15) für weicheres Glühen
            if ti.abs(curr_pos.y) < 0.15:
                r_hit = ti.sqrt(curr_pos.x ** 2 + curr_pos.z ** 2)
                if c.DISK_INNER <= r_hit <= c.DISK_OUTER:

                    # 1. Helligkeitsabfall nach außen
                    intensity = 1.0 / (r_hit - c.DISK_INNER + 0.6)

                    # 2. Relativistisches Beaming (Die linke Seite kommt auf uns zu und strahlt heller)
                    gas_vel = ti.Vector([-curr_pos.z, 0.0, curr_pos.x]).normalized()
                    beaming = -vel.dot(gas_vel)
                    # Der Doppler-Effekt macht eine Seite strahlend hell (Interstellar-Key-Visual)
                    doppler = ti.pow(1.0 + beaming * 0.85, 4.0)

                    # 3. Struktur (Simuliertes Gas-Rauschen)
                    noise = ti.abs(ti.sin(r_hit * 20.0) * ti.cos(ti.atan2(curr_pos.z, curr_pos.x) * 10.0))

                    # 4. Farb-Zusammensetzung (Heißes Orange/Gelb)
                    glow = ti.Vector([1.0, 0.45, 0.15]) * intensity * doppler * (0.4 + 0.6 * noise)

                    # Wir addieren die Farbe für einen volumetrischen Effekt
                    color += glow * 0.2

                    # Wenn der Strahl gesättigt ist, können wir aufhören
                    if color.x > 1.5:
                        break

            # --- B. EVENT HORIZONT ---
            if r < c.HORIZON_FALLBACK:
                color = ti.Vector([0.0, 0.0, 0.0])
                break

            # --- C. HINTERGRUND (STERNENFELD STATT CHECKERBOARD) ---
            if r > 120.0:
                direction = vel.normalized()
                # Mathematische Sterne: Nur sehr kleine Punkte leuchten hell
                star_seed = ti.sin(direction.x * 150.0) * ti.sin(direction.y * 150.0) * ti.sin(direction.z * 150.0)
                if star_seed > 0.992:
                    color += ti.Vector([0.9, 0.9, 1.0]) * star_seed
                break

        pixels[i, j] = color

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