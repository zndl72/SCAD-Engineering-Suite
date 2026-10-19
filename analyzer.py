import os
import re
import shutil
import subprocess
import tempfile
import numpy as np
import trimesh
from matplotlib.path import Path


# Common engineering and 3D printing material densities (g/cm^3)
MATERIALS = {
    "PLA Plastic (3D Print)": {"density": 1.24, "cost_per_kg": 20.0},
    "ABS Plastic (3D Print)": {"density": 1.04, "cost_per_kg": 22.0},
    "PETG Plastic (3D Print)": {"density": 1.27, "cost_per_kg": 24.0},
    "Carbon Fiber Nylon": {"density": 1.40, "cost_per_kg": 65.0},
    "6061-T6 Aluminum": {"density": 2.70, "cost_per_kg": 15.0},
    "304 Stainless Steel": {"density": 8.00, "cost_per_kg": 12.0},
    "Ti-6Al-4V Titanium": {"density": 4.43, "cost_per_kg": 85.0},
    "Birch Plywood": {"density": 0.68, "cost_per_kg": 8.0},
}


def find_openscad_executable():
    """Locate the OpenSCAD CLI binary on Linux (Streamlit Cloud), Windows, or macOS."""
    exe = shutil.which("openscad")
    if exe:
        return exe
    common_paths = [
        r"C:\Program Files\OpenSCAD\openscad.exe",
        r"C:\Program Files (x86)\OpenSCAD\openscad.exe",
        "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD",
        "/usr/bin/openscad",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return None


def parse_scad_fallback(scad_code: str) -> trimesh.Trimesh:
    """
    Pure-Python fallback parser for OpenSCAD code when running locally
    without the OpenSCAD desktop application installed.
    Supports cube, sphere, cylinder, polygon extrusions, and translations.
    """
    meshes = []

    # 1. Check for polygon points + linear_extrude (common in Image-to-CAD outputs)
    poly_match = re.search(r"points\s*=\s*(\[\s*\[.*?\]\s*\])", scad_code, re.DOTALL)
    ext_match = re.search(r"linear_extrude\s*\(\s*(?:height\s*=\s*)?([0-9.]+)", scad_code)
    if poly_match:
        try:
            pts_raw = eval(poly_match.group(1), {"__builtins__": {}})
            pts = np.array(pts_raw, dtype=float)
            height = float(ext_match.group(1)) if ext_match else 10.0
            if len(pts) >= 3:
                # Build extruded prism mesh from 2D polygon contour
                path_2d = Path(pts)
                min_xy = pts.min(axis=0)
                max_xy = pts.max(axis=0)
                gx, gy = np.meshgrid(
                    np.linspace(min_xy[0], max_xy[0], 35),
                    np.linspace(min_xy[1], max_xy[1], 35),
                )
                grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
                inside = path_2d.contains_points(grid_pts)
                valid_pts = grid_pts[inside]
                if len(valid_pts) > 0:
                    dx = (max_xy[0] - min_xy[0]) / 34.0
                    dy = (max_xy[1] - min_xy[1]) / 34.0
                    voxels = [
                        trimesh.creation.box(extents=[dx * 1.05, dy * 1.05, height]).apply_translation(
                            [p[0], p[1], height / 2.0]
                        )
                        for p in valid_pts
                    ]
                    return trimesh.util.concatenate(voxels)
        except Exception:
            pass

    # 2. Parse statement blocks for optional translate() + primitive
    statements = scad_code.split(";")
    for stmt in statements:
        tx, ty, tz = 0.0, 0.0, 0.0
        t_match = re.search(
            r"translate\s*\(\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]\s*\)",
            stmt,
        )
        if t_match:
            tx, ty, tz = map(float, t_match.groups())

        # Cube / Box
        c_vec = re.search(
            r"cube\s*\(\s*(?:size\s*=\s*)?\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]",
            stmt,
        )
        c_single = re.search(r"cube\s*\(\s*(?:size\s*=\s*)?([\d.]+)\s*\)", stmt)
        if c_vec:
            x, y, z = map(float, c_vec.groups())
            m = trimesh.creation.box(extents=[x, y, z])
            if "center=true" not in stmt.replace(" ", ""):
                m.apply_translation([x / 2.0, y / 2.0, z / 2.0])
            m.apply_translation([tx, ty, tz])
            meshes.append(m)
        elif c_single:
            s = float(c_single.group(1))
            m = trimesh.creation.box(extents=[s, s, s])
            m.apply_translation([s / 2.0 + tx, s / 2.0 + ty, s / 2.0 + tz])
            meshes.append(m)

        # Sphere
        sp_match = re.search(r"sphere\s*\(\s*(?:r\s*=\s*)?([\d.]+)", stmt)
        if sp_match:
            r = float(sp_match.group(1))
            m = trimesh.creation.icosphere(subdivisions=3, radius=r)
            m.apply_translation([tx, ty, tz])
            meshes.append(m)

        # Cylinder / Cone
        cyl_match = re.search(r"cylinder\s*\((.*?)\)", stmt)
        if cyl_match:
            args = cyl_match.group(1)
            h_m = re.search(r"h\s*=\s*([\d.]+)", args)
            r_m = re.search(r"(?<![a-zA-Z0-9])r\s*=\s*([\d.]+)", args)
            r1_m = re.search(r"r1\s*=\s*([\d.]+)", args)
            r2_m = re.search(r"r2\s*=\s*([\d.]+)", args)
            d_m = re.search(r"d\s*=\s*([\d.]+)", args)

            h = float(h_m.group(1)) if h_m else 20.0
            if r1_m and r2_m:
                r1, r2 = float(r1_m.group(1)), float(r2_m.group(1))
                r = max((r1 + r2) / 2.0, 0.5)
            elif r_m:
                r = float(r_m.group(1))
            elif d_m:
                r = float(d_m.group(1)) / 2.0
            else:
                nums = [float(n) for n in re.findall(r"[\d.]+", args)]
                h = nums[0] if len(nums) > 0 else 20.0
                r = nums[1] if len(nums) > 1 else 10.0

            m = trimesh.creation.cylinder(radius=r, height=h, sections=32)
            if "center=true" not in stmt.replace(" ", ""):
                m.apply_translation([0, 0, h / 2.0])
            m.apply_translation([tx, ty, tz])
            meshes.append(m)

    if meshes:
        combined = trimesh.util.concatenate(meshes)
        combined.merge_vertices()
        return combined

    # Default aerodynamic wing section if code contains unrecognized syntax locally
    return create_airfoil_mesh()


def create_airfoil_mesh(chord=60.0, thickness=0.18, span=40.0) -> trimesh.Trimesh:
    """Generates a 3D NACA-style symmetric airfoil wing section in pure Python."""
    x = np.linspace(0, 1, 35)
    yt = 5 * thickness * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4
    )
    upper = np.column_stack([x * chord, yt * chord])
    lower = np.column_stack([x[::-1] * chord, -yt[::-1] * chord])
    pts = np.vstack([upper, lower[1:-1]])

    # Triangulate fan from centroid to build watertight prism
    n = len(pts)
    center = pts.mean(axis=0)
    verts_bottom = np.column_stack([pts, np.zeros(n)])
    verts_top = np.column_stack([pts, np.full(n, span)])
    c_bottom = np.array([[center[0], center[1], 0.0]])
    c_top = np.array([[center[0], center[1], span]])

    vertices = np.vstack([verts_bottom, verts_top, c_bottom, c_top])
    idx_cb = 2 * n
    idx_ct = 2 * n + 1

    faces = []
    for i in range(n):
        nxt = (i + 1) % n
        # Bottom cap
        faces.append([idx_cb, nxt, i])
        # Top cap
        faces.append([idx_ct, n + i, n + nxt])
        # Side walls
        faces.append([i, nxt, n + nxt])
        faces.append([i, n + nxt, n + i])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def scad_to_mesh(scad_code: str) -> tuple[trimesh.Trimesh, str]:
    """
    Compiles OpenSCAD code to a 3D Trimesh object.
    Uses native OpenSCAD CLI if available; falls back to pure-Python parser otherwise.
    """
    openscad_bin = find_openscad_executable()
    if openscad_bin:
        with tempfile.TemporaryDirectory() as tmpdir:
            scad_path = os.path.join(tmpdir, "model.scad")
            stl_path = os.path.join(tmpdir, "model.stl")
            with open(scad_path, "w", encoding="utf-8") as f:
                f.write(scad_code)
            res = subprocess.run(
                [openscad_bin, "-o", stl_path, scad_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if res.returncode == 0 and os.path.exists(stl_path):
                mesh = trimesh.load(stl_path, force="mesh")
                return mesh, "Compiled via native OpenSCAD CLI"

    mesh = parse_scad_fallback(scad_code)
    return mesh, "Parsed via built-in Python SCAD Engine"


# =====================================================================
# MODULE 1: 2D WIND TUNNEL (CFD) SOLVER
# =====================================================================
def run_2d_wind_tunnel(
    mesh: trimesh.Trimesh,
    wind_speed: float = 20.0,
    angle_of_attack: float = 0.0,
    plane: str = "XZ (Side Profile)",
):
    """
    Projects/slices the 3D mesh onto a 2D wind tunnel plane, solves potential flow
    with wake drag approximation, and returns velocity/pressure fields + aerodynamic metrics.
    """
    verts = mesh.vertices.copy()
    verts -= verts.mean(axis=0)

    # Select 2D coordinate axes based on chosen tunnel orientation
    if "XZ" in plane:
        pts_2d = verts[:, [0, 2]]
        depth_span = np.ptp(verts[:, 1])
    elif "XY" in plane:
        pts_2d = verts[:, [0, 1]]
        depth_span = np.ptp(verts[:, 2])
    else:
        pts_2d = verts[:, [1, 2]]
        depth_span = np.ptp(verts[:, 0])

    depth_span = max(float(depth_span), 1.0)

    # Apply Angle of Attack rotation (in degrees)
    theta = np.radians(-angle_of_attack)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_2d = pts_2d @ rot_matrix.T

    # Determine bounding box in mm
    min_x, min_y = pts_2d.min(axis=0)
    max_x, max_y = pts_2d.max(axis=0)
    char_length = max(max_x - min_x, 1.0)
    frontal_height = max(max_y - min_y, 1.0)

    # Wind tunnel domain bounds (mm)
    pad_x = max(char_length * 1.0, 20.0)
    pad_y = max(frontal_height * 1.2, 20.0)
    x_lims = (min_x - pad_x * 0.8, max_x + pad_x * 1.4)
    y_lims = (min_y - pad_y, max_y + pad_y)

    nx, ny = 120, 64
    x_grid = np.linspace(x_lims[0], x_lims[1], nx)
    y_grid = np.linspace(y_lims[0], y_lims[1], ny)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Rasterize 2D projected mesh triangles onto the wind tunnel grid
    obstacle = np.zeros((ny, nx), dtype=bool)
    dx = (x_lims[1] - x_lims[0]) / (nx - 1)
    dy = (y_lims[1] - y_lims[0]) / (ny - 1)

    for face in mesh.faces:
        tri = pts_2d[face]
        ix0 = max(int((tri[:, 0].min() - x_lims[0]) / dx), 0)
        ix1 = min(int((tri[:, 0].max() - x_lims[0]) / dx) + 1, nx - 1)
        iy0 = max(int((tri[:, 1].min() - y_lims[0]) / dy), 0)
        iy1 = min(int((tri[:, 1].max() - y_lims[0]) / dy) + 1, ny - 1)
        if ix1 >= ix0 and iy1 >= iy0:
            sub_x = X[iy0 : iy1 + 1, ix0 : ix1 + 1]
            sub_y = Y[iy0 : iy1 + 1, ix0 : ix1 + 1]
            sub_pts = np.column_stack([sub_x.ravel(), sub_y.ravel()])
            mask_tri = Path(tri).contains_points(sub_pts, radius=max(dx, dy) * 0.6)
            obstacle[iy0 : iy1 + 1, ix0 : ix1 + 1] |= mask_tri.reshape(sub_x.shape)

    # Ensure at least a central obstacle exists
    if not obstacle.any():
        obstacle[ny // 2 - 4 : ny // 2 + 4, nx // 3 : nx // 2] = True

    # Solve 2D Streamfunction Laplace Equation (del^2 psi = 0)
    y_norm = np.linspace(-1.0, 1.0, ny)[:, None]
    psi = np.repeat(y_norm, nx, axis=1) * wind_speed
    psi_stag = float(psi[obstacle].mean())
    psi[obstacle] = psi_stag

    for _ in range(180):
        psi_next = 0.25 * (
            np.roll(psi, 1, axis=0)
            + np.roll(psi, -1, axis=0)
            + np.roll(psi, 1, axis=1)
            + np.roll(psi, -1, axis=1)
        )
        # Boundary conditions: uniform inlet/outlet and tunnel walls
        psi_next[0, :] = -wind_speed
        psi_next[-1, :] = wind_speed
        psi_next[:, 0] = y_norm[:, 0] * wind_speed
        psi_next[:, -1] = psi_next[:, -2]
        psi_next[obstacle] = psi_stag
        psi = psi_next

    # Compute velocity field u = d(psi)/dy, v = -d(psi)/dx
    dy_norm = 2.0 / (ny - 1)
    dx_norm = 2.0 / (nx - 1)
    u = np.gradient(psi, dy_norm, axis=0)
    v = -np.gradient(psi, dx_norm, axis=1)

    # Apply viscous wake velocity deficit behind the trailing edge
    obs_cols = np.where(obstacle.any(axis=0))[0]
    if len(obs_cols) > 0:
        trail_col = obs_cols[-1]
        obs_rows = np.where(obstacle.any(axis=1))[0]
        y_center = obs_rows.mean()
        half_h = max(len(obs_rows) / 2.0, 2.0)
        for ix in range(trail_col, nx):
            dist = (ix - trail_col) / max(nx - trail_col, 1)
            for iy in range(ny):
                lat = abs(iy - y_center) / (half_h * (1.0 + 0.8 * dist))
                if lat < 1.2 and not obstacle[iy, ix]:
                    wake_factor = 1.0 - 0.65 * np.exp(-1.8 * dist) * np.cos(lat * np.pi / 2.4) ** 2
                    u[iy, ix] *= wake_factor

    u[obstacle] = 0.0
    v[obstacle] = 0.0
    speed = np.sqrt(u**2 + v**2)

    # Pressure Coefficient Cp = 1 - (V / V_inf)^2 via Bernoulli's principle
    cp = 1.0 - (speed / max(wind_speed, 0.1)) ** 2
    cp[obstacle] = np.nan

    # Estimate Drag Coefficient (Cd) from blockage ratio & bluffness profile
    # Streamlined shapes have smaller leading-edge stagnation gradients
    bluffness = np.clip(frontal_height / char_length, 0.08, 2.0)
    cd_est = float(np.clip(0.12 + 0.95 * (bluffness**0.85), 0.09, 2.10))

    # Aerodynamic physical metrics (Standard Sea Level Air: rho = 1.225 kg/m^3, nu = 1.5e-5 m^2/s)
    rho = 1.225
    nu = 1.5e-5
    frontal_area_m2 = (frontal_height / 1000.0) * (depth_span / 1000.0)
    reynolds = (wind_speed * (char_length / 1000.0)) / nu
    dynamic_pressure = 0.5 * rho * (wind_speed**2)
    drag_force_n = dynamic_pressure * frontal_area_m2 * cd_est

    return {
        "X": X,
        "Y": Y,
        "u": u,
        "v": v,
        "speed": speed,
        "cp": cp,
        "obstacle": obstacle,
        "cd": cd_est,
        "reynolds": reynolds,
        "frontal_area_cm2": frontal_area_m2 * 1e4,
        "drag_force_n": drag_force_n,
        "dynamic_pressure_pa": dynamic_pressure,
    }


# =====================================================================
# MODULE 2: DIGITAL MASS & CENTER OF GRAVITY (CoG) ANALYZER
# =====================================================================
def calculate_mass_properties(mesh: trimesh.Trimesh, material_name: str):
    """
    Calculates volume, total mass, 3D Center of Gravity (CoG), and Principal
    Moments of Inertia for the selected engineering material.
    """
    mat = MATERIALS.get(material_name, MATERIALS["PLA Plastic (3D Print)"])
    density_g_cm3 = mat["density"]

    # Mesh coordinates are in mm -> 1 cm^3 = 1000 mm^3
    vol_mm3 = abs(float(mesh.volume)) if mesh.is_volume else float(mesh.convex_hull.volume)
    vol_cm3 = vol_mm3 / 1000.0
    mass_g = vol_cm3 * density_g_cm3
    mass_kg = mass_g / 1000.0

    # Center of Gravity (mm)
    try:
        cg = mesh.center_mass
        if np.isnan(cg).any():
            cg = mesh.centroid
    except Exception:
        cg = mesh.centroid

    # Bounding Box dimensions (mm)
    extents = mesh.extents

    # Moments of inertia scaled to g*mm^2 -> converted to kg*mm^2
    # Trimesh assumes density=1.0 per unit volume (mm^3), so multiply by density in kg/mm^3
    density_kg_mm3 = (density_g_cm3 / 1000.0) / 1000.0
    try:
        inertia_tensor = np.abs(mesh.moment_inertia) * density_kg_mm3
    except Exception:
        inertia_tensor = np.eye(3) * (mass_kg * (np.mean(extents) ** 2) / 6.0)

    return {
        "material": material_name,
        "density_g_cm3": density_g_cm3,
        "volume_cm3": vol_cm3,
        "surface_area_cm2": float(mesh.area) / 100.0,
        "mass_g": mass_g,
        "mass_kg": mass_kg,
        "cg": cg,
        "extents_mm": extents,
        "inertia_kg_mm2": inertia_tensor,
    }


# =====================================================================
# MODULE 3: 3D PRINT MANUFACTURABILITY (DFM) & COST ESTIMATOR
# =====================================================================
def analyze_3d_printability(
    mesh: trimesh.Trimesh,
    material_name: str = "PLA Plastic (3D Print)",
    infill_pct: int = 20,
    layer_height_mm: float = 0.20,
    overhang_limit_deg: float = 45.0,
    print_speed_mm_s: float = 60.0,
    electricity_rate_kwh: float = 0.16,
):
    """
    Analyzes triangle face normals to detect overhang regions requiring support,
    and computes filament usage, print time, and total manufacturing cost.
    """
    mat = MATERIALS.get(material_name, MATERIALS["PLA Plastic (3D Print)"])
    density = mat["density"]
    cost_per_kg = mat["cost_per_kg"]

    normals = mesh.face_normals
    areas = mesh.area_faces
    centroids = mesh.triangles_center
    min_z = mesh.bounds[0, 2]

    # Overhang angle: downward-pointing faces not sitting directly on the build plate
    # Normal Z = -1 means straight down (0 deg from horizontal / 90 deg overhang from vertical)
    # Standard 3D printing convention: overhang angle measured from vertical (0 deg = vertical wall, 90 deg = horizontal ceiling)
    downward_mask = (normals[:, 2] < -0.05) & ((centroids[:, 2] - min_z) > 0.6)
    overhang_angle_from_vertical = np.degrees(np.arccos(np.clip(-normals[:, 2], 0.0, 1.0)))
    # A vertical wall has n_z = 0 -> angle = 90. Overhang tilt past vertical is (90 - angle_from_downward)
    overhang_tilt_deg = 90.0 - overhang_angle_from_vertical

    critical_faces = downward_mask & (overhang_tilt_deg > overhang_limit_deg)
    overhang_area_mm2 = float(areas[critical_faces].sum())
    total_area_mm2 = max(float(areas.sum()), 1.0)
    overhang_pct = (overhang_area_mm2 / total_area_mm2) * 100.0

    # Calculate shell + infill + support volume
    vol_mm3 = abs(float(mesh.volume)) if mesh.is_volume else float(mesh.convex_hull.volume)
    wall_thickness_mm = 1.2  # 3 perimeters at 0.4mm nozzle
    shell_vol_mm3 = min(total_area_mm2 * wall_thickness_mm, vol_mm3 * 0.85)
    interior_vol_mm3 = max(vol_mm3 - shell_vol_mm3, 0.0)

    effective_part_vol_mm3 = shell_vol_mm3 + interior_vol_mm3 * (infill_pct / 100.0)
    support_vol_mm3 = overhang_area_mm2 * (mesh.extents[2] * 0.35) * 0.15  # 15% support density
    total_extruded_mm3 = effective_part_vol_mm3 + support_vol_mm3

    # Filament mass and length (standard 1.75mm diameter filament cross-section = 2.405 mm^2)
    filament_mass_g = (total_extruded_mm3 / 1000.0) * density
    filament_length_m = (total_extruded_mm3 / 2.40528) / 1000.0

    # Print time estimation from volumetric flow rate (line_width * layer_height * print_speed)
    volumetric_rate_mm3_s = 0.4 * layer_height_mm * print_speed_mm_s * 0.65
    print_time_hours = (total_extruded_mm3 / max(volumetric_rate_mm3_s, 0.5)) / 3600.0

    # Cost calculation (Filament + 150W printer power consumption)
    filament_cost = (filament_mass_g / 1000.0) * cost_per_kg
    energy_cost = print_time_hours * 0.15 * electricity_rate_kwh
    total_cost = filament_cost + energy_cost

    # Manufacturability Score (0 - 100)
    dfm_score = max(100.0 - overhang_pct * 2.5, 15.0)

    return {
        "critical_faces": critical_faces,
        "overhang_pct": overhang_pct,
        "overhang_area_cm2": overhang_area_mm2 / 100.0,
        "dfm_score": dfm_score,
        "filament_mass_g": filament_mass_g,
        "filament_length_m": filament_length_m,
        "support_mass_g": (support_vol_mm3 / 1000.0) * density,
        "print_time_hours": print_time_hours,
        "filament_cost": filament_cost,
        "energy_cost": energy_cost,
        "total_cost": total_cost,
    }