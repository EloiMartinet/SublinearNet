import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.cm as cm
import numpy as np
import os
import shutil
import csv
import matplotlib
import torch
from matplotlib import patches
from matplotlib.path import Path
import pyvista as pv
import tempfile
import matplotlib.tri as mtri

from shapes.invertible_nn import ConvexDiffeo

# fully non-interactive backend. Solved memory issues
matplotlib.use('Agg')
pv.OFF_SCREEN = True


def plot_shape(model, output, n_points=500, video=False, plot_mesh=False, color_fn=None, background_fn=None):
    if model.dim == 2:
        plot_shape_2d(model, output, n_points, color_fn=color_fn, background_fn=background_fn)
    if model.dim == 3 and not video:
        plot_shape_3d(model, output, n_points, color_fn=color_fn)
    if model.dim == 3 and video:
        plot_shape_3d_movie(model, output, n_points)
    if plot_mesh:
        from skfem_mesh_utils import create_2d_moved_mesh_mmg
        import skfem as fem
        from skfem.visuals.matplotlib import draw

        with tempfile.TemporaryDirectory() as tmpdir:

            mesh_file = os.path.join(tmpdir, "boundary.mesh")
            create_2d_moved_mesh_mmg(
                model,
                rel_mesh_size=model.mesh_length,
                output=mesh_file
            )
            moved_mesh = fem.MeshTri.load(mesh_file)
            draw(moved_mesh)
            plt.savefig(output)
            plt.close()

def plot_shape_2d(model, output, n_points=200, color_fn=None, background_fn=None):
    """_summary_

    Args:
        model (_type_): _description_
        output (_type_): _description_
        n_points (int, optional): _description_. Defaults to 500.
        color_fn (_type_, optional): The coloring function. Warning: it is defined on the reference domain! Defaults to None.
    """
    # Get dtype/device
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    # Parametric boundary
    theta = torch.linspace(0, 2*torch.pi, n_points, device=model_device, dtype=model_dtype)
    x, y = torch.cos(theta), torch.sin(theta)
    p = torch.stack([x, y]).T

    with torch.no_grad():
        p = model(p).detach().cpu().numpy()

    fig, ax = plt.subplots()
    
    # --------------------------------------------------
    # BACKGROUND FUNCTION
    # --------------------------------------------------
    if background_fn is not None:
        # Compute bounding square
        xmin, ymin = p.min(axis=0) - 0.1
        xmax, ymax = p.max(axis=0) + 0.1

        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        half_size = 0.5 * max(xmax - xmin, ymax - ymin)

        xmin, xmax = cx - half_size, cx + half_size
        ymin, ymax = cy - half_size, cy + half_size

        # Create grid
        grid_n = n_points
        xs = np.linspace(xmin, xmax, grid_n)
        ys = np.linspace(ymin, ymax, grid_n)
        X, Y = np.meshgrid(xs, ys)

        grid = np.stack([X.ravel(), Y.ravel()], axis=1)
        grid_torch = torch.tensor(grid, device=model_device, dtype=model_dtype)

        with torch.no_grad():
            Z = background_fn(grid_torch).detach().cpu().numpy()

        Z = Z.reshape(grid_n, grid_n)

        # Plot background
        im = ax.imshow(
            Z,
            extent=(xmin, xmax, ymin, ymax),
            origin='lower',
            cmap='viridis',
            alpha=0.7,
        )

        # 0 level set
        ax.contour(
            X,
            Y,
            Z,
            levels=[0],
            colors='purple',
            linewidths=1,
            linestyles='--',
        )
        fig.colorbar(im, ax=ax)
    
    if color_fn is not None:
        # --- Sample interior points and compute color ---
        n_samples = n_points * n_points
        pts = model.sample_ball(n_samples)
        pts_forward = model(pts).detach().cpu().numpy()
        values = color_fn(pts).detach().cpu().numpy()
        pts = pts.detach().cpu().numpy()

        # Triangulation
        tri = mtri.Triangulation(pts_forward[:, 0], pts_forward[:, 1])

        # Mask triangles outside polygon
        tri_centers = pts_forward[tri.triangles].mean(axis=1)

        # Plot colored surface
        tpc = ax.tripcolor(tri, values, shading='gouraud', cmap='rainbow')

        # Add colorbar
        fig.colorbar(tpc, ax=ax)

    if color_fn is None and background_fn is None:
        # Simple polygon
        polygon = patches.Polygon(
            p, closed=True, fill=True, edgecolor='black', linewidth=1
        )
        ax.add_patch(polygon)
    else:
        # Draw boundary outline
        ax.plot(p[:, 0], p[:, 1], color='black', linewidth=2)

    ax.set_aspect('equal', 'box')
    ax.axis('off')
    ax.autoscale_view()

    plt.savefig(output, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_shape_3d(model, output, n_points=1000, window_size=[800, 800], color_fn=None):
    import pyvista as pv
    import torch

    try:
        param = next(model.parameters())
        model_dtype = param.dtype
        model_device = param.device
    except StopIteration:
        model_dtype = torch.float32
        model_device = torch.device("cpu")

    # Sample unit sphere
    sphere = pv.Sphere(radius=1.0, theta_resolution=n_points, phi_resolution=n_points)
    points = torch.tensor(sphere.points, dtype=model_dtype, device=model_device)

    with torch.no_grad():
        mapped_points = model(points).detach().cpu().numpy()

    phi_mesh = pv.PolyData(mapped_points, sphere.faces)

    # --- Coloring ---
    if color_fn is not None:
        values = color_fn(points).detach().cpu().numpy()
        # values = (values - values.min()) / (values.max() - values.min() + 1e-8)
        phi_mesh["values"] = values
        scalars_name = "values"
        show_bar = True
    else:
        z_vals = mapped_points[:, 2]
        z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-8)
        phi_mesh["values"] = z_norm
        scalars_name = "values"
        show_bar = False

    # Plotter
    plotter = pv.Plotter(off_screen=True, window_size=window_size)

    plotter.add_mesh(
        phi_mesh,
        scalars=scalars_name,
        cmap="plasma",
        smooth_shading=True,
        specular=0.6,
        ambient=0.3,
        diffuse=0.6,
        show_scalar_bar=show_bar,
    )

    # Lights
    plotter.add_light(pv.Light(position=(5,5,5), focal_point=(0,0,0), intensity=0.1))
    plotter.add_light(pv.Light(position=(-3,-3,2), focal_point=(0,0,0), intensity=0.2))

    # Camera
    camera_pos = np.array([5.0, 3.0, 3.0])
    focal_point = np.array([0.0, 0.0, 0.0])

    plotter.camera_position = [
        camera_pos.tolist(),
        focal_point.tolist(),
        (0, 0, 1),
    ]
    plotter.show(screenshot=output)
    plotter.close()

def plot_shape_3d_movie(model, output, n_points=100, window_size=[800, 800]):
    # Get the dtype and device of the model
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    # Sample unit sphere
    sphere = pv.Sphere(radius=1.0, theta_resolution=n_points, phi_resolution=n_points)
    points = torch.tensor(sphere.points, dtype=model_dtype, device=model_device)

    with torch.no_grad():
        mapped_points = model(points).numpy()

    phi_mesh = pv.PolyData(mapped_points, sphere.faces)

    # Off-screen plotter and styling
    plotter = pv.Plotter(off_screen=True, window_size=window_size)

    # Subtle color by height
    z_vals = mapped_points[:, 2]
    z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())
    phi_mesh["z"] = z_norm

    plotter.add_mesh(
        phi_mesh,
        scalars="z",
        cmap="plasma",      # muted, subtle colormap
        smooth_shading=True,
        specular=0.6,
        ambient=0.3,
        diffuse=0.6,
        show_scalar_bar=False,
    )

    # Lights
    plotter.add_light(pv.Light(position=(5,5,5), focal_point=(0,0,0), intensity=0.1))
    plotter.add_light(pv.Light(position=(-3,-3,2), focal_point=(0,0,0), intensity=0.2))

    # Setup movie
    plotter.open_movie(output, framerate=30)  # output file

    n_frames = 120  # 4 seconds at 30fps

    for i in range(n_frames):
        # rotate camera around z-axis
        plotter.camera.Azimuth(360 / n_frames)
        plotter.render()            # render current frame
        plotter.write_frame()       # add frame to video

    plotter.close()




def plot_point_cloud_3d(points, output, window_size=(800, 800)):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    cloud = pv.PolyData(points)

    plotter = pv.Plotter(off_screen=True, window_size=window_size)

    # --- Camera setup (fixed reference) ---
    camera_pos = np.array([5.0, 3.0, 3.0])
    focal_point = np.array([0.0, 0.0, 0.0])

    plotter.camera_position = [
        camera_pos.tolist(),
        focal_point.tolist(),
        (0, 0, 1),
    ]

    # --- CAMERA DISTANCE (fog / alpha) ---
    cam_dist = np.linalg.norm(points - camera_pos, axis=1)
    cam_norm = (cam_dist - cam_dist.min()) / (cam_dist.max() - cam_dist.min() + 1e-8)

    alpha = 1 - cam_norm # np.exp(-2.5 * cam_norm)

    # --- CENTER DISTANCE (coloring) ---
    center_dist = np.linalg.norm(points, axis=1)
    cen_norm = (center_dist - center_dist.min()) / (center_dist.max() - center_dist.min() + 1e-8)

    # Apply colormap
    colormap = cm.get_cmap("summer")
    rgb = colormap(cen_norm)[:, :3]  # drop alpha from colormap

    # --- Combine into RGBA ---
    colors = np.zeros((points.shape[0], 4))
    colors[:, :3] = rgb
    colors[:, 3] = alpha

    cloud["colors"] = colors

    plotter.add_points(
        cloud,
        scalars="colors",
        rgba=True,
        render_points_as_spheres=True,
        point_size=10,
    )

    plotter.add_light(pv.Light(position=(5, 5, 5), focal_point=(0, 0, 0), intensity=0.2))

    plotter.show(screenshot=output)
    plotter.close()

if __name__ == '__main__':
    model = ConvexDiffeo(mesh_file='tmp.msh', input_size=2, n_unit=50)
    plot_shape(model, output="coucou.png")

