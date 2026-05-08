## 1. Project Architecture Overview

The project will be divided into four distinct modules. Each module has a specific responsibility, preventing "spaghetti code."

### Module A: The Spacetime Engine (`engine.py`)

This is the "Math Heart." It handles the Kerr Metric and the Runge-Kutta integration.

* **Physics Rules:** Defines the mass ($M$) and spin ($a$) of the black hole.
* **Geodesic Solver:** A Taichi kernel that calculates the next position of a photon.
* **Boundary Conditions:** Detects if a ray is "captured" (hits the horizon) or "escaped" (hits the skybox).

### Module B: The Scene & Camera (`camera.py`)

Handles the perspective and user interaction.

* **Ray Initialization:** Converts 2D pixel coordinates $(u, v)$ into 3D momentum vectors $p_\mu$.
* **Transformations:** Handles camera rotation, zoom, and tilt.

### Module C: The Material & Light (`shading.py`)

This makes it "cool." It calculates colors without touching the physics.

* **Accretion Disk Logic:** Defines the texture, density, and temperature of the gas disk.
* **Relativistic Effects:** Applies the Doppler boost (brightness shift) and Gravitational Redshift.
* **Skybox:** Samples a high-resolution Milky Way texture for the background stars.

### Module D: The Orchestrator (`main.py`)

The entry point that connects the modules, handles the Taichi initialization, and manages the video export (using `ffmpeg`).

---

## 2. Project Rules & Standards

To keep this "clean" for a beginner, we will enforce three strict development rules:

1. **Unit Consistency:** All calculations must use **Geometrizied Units** ($G = c = 1$). This simplifies the math drastically—for example, the Schwarzschild radius becomes $R_s = 2M$.
2. **No Magic Numbers:** All physics constants (Spin, Disk Radius, Camera FOV) must live in a `config.yaml` or a dedicated `Constants` class.
3. **Kernel/Python Separation:** Code inside a `@ti.kernel` must be strictly mathematical. Use standard Python only for UI and file saving.

---

## 3. The Development Roadmap (The "Sprints")

| Phase | Task | Deliverable |
| --- | --- | --- |
| **Phase 1** | **The Void** | A black sphere (event horizon) against a solid color background using simple straight rays. |
| **Phase 2** | **The Warp** | Implementing RK4 integration to make background stars "bend" around the sphere (Gravitational Lensing). |
| **Phase 3** | **The Fire** | Adding the Accretion Disk plane and basic color mapping based on distance from center. |
| **Phase 4** | **The Relativity** | Adding Doppler shifting (left side bright, right side dim) and Kerr (spinning) metrics. |
| **Phase 5** | **The Render** | Implementing the video sequence generator to animate camera orbits. |

---

## 4. Folder Structure (The "Clean Build")

```text
EventHorizon-RT/
├── core/
│   ├── integrator.py    # Geodesic math (Taichi)
│   ├── metrics.py       # Schwarzschild vs Kerr equations
│   └── constants.py     # G=c=1 definitions
├── scene/
│   ├── camera.py        # Ray generation
│   └── disk.py          # Accretion disk properties
├── utils/
│   ├── loaders.py       # Texture/Config loading
│   └── post_process.py  # Bloom & Tone mapping
├── config.yaml          # Resolution, Spin, Mass settings
└── main.py              # The main loop

```

---

## 5. Why this works for a Beginner

* **Visual Feedback:** You see every math error immediately (e.g., if the rays don't bend, the image looks "flat").
* **Modular Learning:** You can master "Camera Rays" before ever touching "General Relativity."
* **Performance:** Using Taichi means you don't have to learn C++ or CUDA kernels manually, but you get the same speed.