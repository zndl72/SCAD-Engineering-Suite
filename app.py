import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from analyzer import (
    MATERIALS,
    scad_to_mesh,
    run_2d_wind_tunnel,
    calculate_mass_properties,
    analyze_3d_printability,
)

st.set_page_config(
    page_title="SCAD Engineering & CFD Suite",
    page_icon="⚙️",
    layout="wide",
)

# Built-in SCAD templates for instant testing
PRESETS = {
    "Custom / Paste from ImageToCAD-App": """// Paste your OpenSCAD code from ImageToCAD-App here!
// Example: Stepped Aerodynamic Drone Fuselage Pod
translate([0, 0, 0]) cube([60, 24, 14]);
translate([12, 2, 14]) cube([36, 20, 10]);
translate([60, 12, 7]) sphere(r=12);
""",
    "NACA Aerodynamic Wing Section": """// Symmetrical Airfoil Polygon Extrusion
linear_extrude(height = 40)
polygon(points=[
    [0,0], [5,4.2], [15,6.8], [30,7.2], [45,5.4],
    [60,2.5], [75,0], [60,-2.5], [45,-5.4], [30,-7.2],
    [15,-6.8], [5,-4.2]
]);
""",
    "Cantilever Overhang Test Bracket": """// Structural Bracket with 3D Print Overhang Bridge
cube([20, 25, 45]);
translate([0, 0, 35]) cube([55, 25, 10]);
translate([45, 12.5, 25]) cylinder(h=10, r=8);
""",
}

st.title("⚙️ OpenSCAD Engineering Analysis Suite")
st.markdown(
    "Import `.scad` code directly from **ImageToCAD-App** to run a **2D Wind Tunnel CFD Simulation**, "
    "compute **3D Mass & Center of Gravity**, and evaluate **3D Print Manufacturability & Cost**."
)

# =====================================================================
# SIDEBAR: INPUT CAD & MATERIAL SELECTION
# =====================================================================
with st.sidebar:
    st.header("1. CAD Geometry Input")
    input_mode = st.radio("Input Source:", ["Paste OpenSCAD Code", "Upload .scad / .stl File"])

    mesh = None
    status_msg = ""

    if input_mode == "Paste OpenSCAD Code":
        preset_choice = st.selectbox("Load Preset or Custom Code:", list(PRESETS.keys()))
        scad_code = st.text_area("OpenSCAD (.scad) Code:", value=PRESETS[preset_choice], height=220)
        if scad_code.strip():
            with st.spinner("Generating 3D Mesh from SCAD..."):
                mesh, status_msg = scad_to_mesh(scad_code)
    else:
        uploaded = st.file_uploader("Upload .scad or .stl", type=["scad", "stl"])
        if uploaded:
            if uploaded.name.endswith(".scad"):
                scad_code = uploaded.read().decode("utf-8")
                mesh, status_msg = scad_to_mesh(scad_code)
            else:
                import trimesh
                mesh = trimesh.load(uploaded, file_type="stl", force="mesh")
                status_msg = "Loaded directly from STL file"

    st.divider()
    st.header("2. Global Material Selection")
    selected_material = st.selectbox("Select Engineering Material:", list(MATERIALS.keys()))

if mesh is None:
    st.info("👈 Paste your OpenSCAD code or upload a CAD file in the sidebar to begin analysis.")
    st.stop()

st.caption(f"✅ **Mesh Status:** {status_msg} | **Vertices:** {len(mesh.vertices):,} | **Faces:** {len(mesh.faces):,}")

# Create the 3 primary engineering tabs
tab_cfd, tab_mass, tab_dfm = st.tabs([
    "🌬️ 1. 2D Wind Tunnel (CFD)",
    "⚖️ 2. Digital Mass & Center of Gravity",
    "🖨️ 3. 3D Print Manufacturability & Cost",
])

# =====================================================================
# TAB 1: 2D WIND TUNNEL (CFD)
# =====================================================================
with tab_cfd:
    st.subheader("2D Computational Fluid Dynamics (CFD) Wind Tunnel")
    col_ctrl, col_plot = st.columns([1, 3])

    with col_ctrl:
        plane = st.selectbox(
            "Cross-Section Orientation:",
            ["XZ (Side Profile)", "XY (Top-Down Profile)", "YZ (Front Profile)"],
        )
        wind_speed = st.slider("Inlet Wind Velocity (m/s):", min_value=1.0, max_value=60.0, value=20.0, step=1.0)
        aoa = st.slider("Angle of Attack (°):", min_value=-30.0, max_value=30.0, value=0.0, step=1.0)
        field_mode = st.radio("Contour Field Overlay:", ["Velocity Magnitude (m/s)", "Pressure Coefficient (Cp)"])

    cfd = run_2d_wind_tunnel(mesh, wind_speed=wind_speed, angle_of_attack=aoa, plane=plane)

    # Top Telemetry Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Est. Drag Coefficient (Cd)", f"{cfd['cd']:.2f}")
    m2.metric("Aerodynamic Drag Force", f"{cfd['drag_force_n']:.3f} N")
    m3.metric("Reynolds Number (Re)", f"{cfd['reynolds']:,.0f}")
    m4.metric("Dynamic Pressure", f"{cfd['dynamic_pressure_pa']:.1f} Pa")

    with col_plot:
        fig, ax = plt.subplots(figsize=(10, 4.8), dpi=120)
        X, Y = cfd["X"], cfd["Y"]

        if "Velocity" in field_mode:
            im = ax.pcolormesh(X, Y, cfd["speed"], shading="auto", cmap="turbo")
            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label("Flow Speed (m/s)")
        else:
            im = ax.pcolormesh(X, Y, cfd["cp"], shading="auto", cmap="coolwarm", vmin=-1.5, vmax=1.0)
            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label("Pressure Coefficient (Cp)")

        # Overlay streamlines
        ax.streamplot(
            X, Y, cfd["u"], cfd["v"],
            color="white", linewidth=0.85, density=1.3, arrowsize=0.9
        )

        # Render solid CAD cross-section obstacle
        obs_rgba = np.zeros((*cfd["obstacle"].shape, 4))
        obs_rgba[cfd["obstacle"]] = [0.18, 0.20, 0.25, 1.0]
        ax.imshow(
            obs_rgba,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            aspect="auto",
            zorder=5,
        )

        ax.set_title(f"Wind Tunnel Flow Streamlines ({plane}, AoA = {aoa}°)")
        ax.set_xlabel("Tunnel Streamwise Position (mm)")
        ax.set_ylabel("Tunnel Vertical Position (mm)")
        fig.tight_layout()
        st.pyplot(fig)


# =====================================================================
# TAB 2: DIGITAL MASS & CENTER OF GRAVITY
# =====================================================================
with tab_mass:
    st.subheader("Volumetric Mass, Center of Gravity (CoG) & Moments of Inertia")
    props = calculate_mass_properties(mesh, selected_material)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Mass", f"{props['mass_g']:.2f} g", f"{props['mass_kg']:.4f} kg")
    c2.metric("Volume", f"{props['volume_cm3']:.2f} cm³")
    c3.metric("Surface Area", f"{props['surface_area_cm2']:.2f} cm²")
    c4.metric("Material Density", f"{props['density_g_cm3']:.2f} g/cm³")

    col_3d, col_info = st.columns([2, 1])

    with col_3d:
        v = mesh.vertices
        f = mesh.faces
        cg = props["cg"]
        bounds = mesh.bounds

        fig_3d = go.Figure()

        # Translucent 3D Mesh so internal CoG marker is clearly visible
        fig_3d.add_trace(
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                opacity=0.40,
                color="#3b82f6",
                name="CAD Geometry",
                hoverinfo="skip",
            )
        )

        # Center of Gravity Marker
        fig_3d.add_trace(
            go.Scatter3d(
                x=[cg[0]], y=[cg[1]], z=[cg[2]],
                mode="markers+text",
                marker=dict(size=9, color="#ef4444", symbol="diamond"),
                text=[f"CoG ({cg[0]:.1f}, {cg[1]:.1f}, {cg[2]:.1f}) mm"],
                textposition="top center",
                name="Center of Gravity",
            )
        )

        # 3D Axis Crosshairs intersecting at the CoG
        for axis_idx, axis_name in enumerate(["X", "Y", "Z"]):
            pts_line = np.tile(cg, (2, 1))
            pts_line[0, axis_idx] = bounds[0, axis_idx] - 5.0
            pts_line[1, axis_idx] = bounds[1, axis_idx] + 5.0
            fig_3d.add_trace(
                go.Scatter3d(
                    x=pts_line[:, 0], y=pts_line[:, 1], z=pts_line[:, 2],
                    mode="lines",
                    line=dict(color="#ef4444", width=3, dash="dash"),
                    showlegend=False,
                )
            )

        fig_3d.update_layout(
            height=480,
            margin=dict(l=0, r=0, t=30, b=0),
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                aspectmode="data",
            ),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_info:
        st.markdown("#### 📍 Center of Gravity Coordinates")
        st.write(f"- **X_cg:** `{cg[0]:.2f} mm`")
        st.write(f"- **Y_cg:** `{cg[1]:.2f} mm`")
        st.write(f"- **Z_cg:** `{cg[2]:.2f} mm`")

        st.markdown("#### 📐 Bounding Box Dimensions")
        ext = props["extents_mm"]
        st.write(f"- **Length (X):** `{ext[0]:.2f} mm`")
        st.write(f"- **Width (Y):** `{ext[1]:.2f} mm`")
        st.write(f"- **Height (Z):** `{ext[2]:.2f} mm`")

        st.markdown("#### 🔄 Inertia Tensor (kg·mm²)")
        st.dataframe(
            props["inertia_kg_mm2"],
            column_config={"0": "I_x", "1": "I_y", "2": "I_z"},
            use_container_width=True,
        )


# =====================================================================
# TAB 3: 3D PRINT MANUFACTURABILITY & COST ESTIMATOR
# =====================================================================
with tab_dfm:
    st.subheader("3D Print Overhang Analysis (DFM) & Cost Estimator")

    col_settings, col_dfm_view = st.columns([1, 2])

    with col_settings:
        infill = st.slider("Infill Density (%):", 5, 100, 20, step=5)
        layer_h = st.select_slider("Layer Height (mm):", options=[0.12, 0.16, 0.20, 0.28], value=0.20)
        overhang_deg = st.slider("Max Un-supported Overhang Angle (°):", 30, 65, 45)
        speed = st.slider("Print Speed (mm/s):", 30, 150, 60, step=10)
        elec_rate = st.number_input("Electricity Rate ($/kWh):", value=0.16, step=0.02)

    dfm = analyze_3d_printability(
        mesh,
        material_name=selected_material,
        infill_pct=infill,
        layer_height_mm=layer_h,
        overhang_limit_deg=overhang_deg,
        print_speed_mm_s=speed,
        electricity_rate_kwh=elec_rate,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Estimated Cost", f"${dfm['total_cost']:.2f}")
    k2.metric("Filament Required", f"{dfm['filament_mass_g']:.1f} g", f"{dfm['filament_length_m']:.2f} m")
    hrs = int(dfm["print_time_hours"])
    mins = int((dfm["print_time_hours"] - hrs) * 60)
    k3.metric("Estimated Print Time", f"{hrs}h {mins}m")
    k4.metric("Manufacturability Score", f"{dfm['dfm_score']:.0f} / 100")

    with col_dfm_view:
        v = mesh.vertices
        f = mesh.faces
        face_colors = [
            "#ef4444" if is_crit else "#94a3b8" for is_crit in dfm["critical_faces"]
        ]

        fig_dfm = go.Figure(
            data=[
                go.Mesh3d(
                    x=v[:, 0], y=v[:, 1], z=v[:, 2],
                    i=f[:, 0], j=f[:, 1], k=f[:, 2],
                    facecolor=face_colors,
                    opacity=1.0,
                    name="Overhang Map",
                )
            ]
        )
        fig_dfm.update_layout(
            title="3D Overhang Inspection (Red Faces = Support Material Required)",
            height=450,
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z ( Build Direction, mm)",
                aspectmode="data",
            ),
        )
        st.plotly_chart(fig_dfm, use_container_width=True)

        st.info(
            f"🔍 **DFM Summary:** `{dfm['overhang_pct']:.1f}%` of total surface area "
            f"(`{dfm['overhang_area_cm2']:.2f} cm²`) exceeds the `{overhang_deg}°` overhang threshold "
            f"and requires an estimated `{dfm['support_mass_g']:.1f} g` of breakaway support material."
        )