import ufl
from mpi4py import MPI
from dolfinx import io, mesh
import gmsh
# def build_two_region_ball_mesh(R_outer: float = 1.0,
#                                R_inner: float = 0.4,
#                                lc_outer: float = 0.12,
#                                lc_inner: float = 0.08):
    
#     gmsh.initialize()
#     gmsh.model.add("cell_with_nucleus_3d")

#     v_outer = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, R_outer)
#     v_inner = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, R_inner)

#     out, _ = gmsh.model.occ.fragment([(3, v_outer)], [(3, v_inner)])
#     gmsh.model.occ.synchronize()

#     volumes = [ent for ent in out if ent[0] == 3]

#     def bbox_radius(vol_tag):
#         xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(3, vol_tag)
#         return max(abs(xmax), abs(ymax), abs(zmax))

#     v_small, v_large = sorted([v[1] for v in volumes], key=bbox_radius)
#     inner_vol = v_small
#     outer_vol = v_large

#     gmsh.model.addPhysicalGroup(3, [outer_vol], tag=1)  # Omega_c
#     gmsh.model.setPhysicalName(3, 1, "Omega_c")
#     gmsh.model.addPhysicalGroup(3, [inner_vol], tag=2)  # Omega_n
#     gmsh.model.setPhysicalName(3, 2, "Omega_n")

#     bnd_outer = gmsh.model.getBoundary([(3, outer_vol)], oriented=False, recursive=False)
#     bnd_inner = gmsh.model.getBoundary([(3, inner_vol)], oriented=False, recursive=False)

#     surf_outer = [s[1] for s in bnd_outer if s[0] == 2]
#     surf_inner = [s[1] for s in bnd_inner if s[0] == 2]

#     def surface_radius(surf_tag):
#         xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, surf_tag)
#         return max(abs(xmax), abs(ymax), abs(zmax))

#     outer_surface = max(surf_outer, key=surface_radius)
#     interface_surface = max(surf_inner, key=surface_radius)

#     gmsh.model.addPhysicalGroup(2, [outer_surface], tag=11)  # Gamma_c
#     gmsh.model.setPhysicalName(2, 11, "Gamma_c")
#     gmsh.model.addPhysicalGroup(2, [interface_surface], tag=12)  # Gamma_n
#     gmsh.model.setPhysicalName(2, 12, "Gamma_n")

#     gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc_outer)
#     pts_on_iface = gmsh.model.getBoundary([(2, interface_surface)], oriented=False, recursive=True)
#     pts_on_iface = [p for p in pts_on_iface if p[0] == 0]
#     if pts_on_iface:
#         gmsh.model.mesh.setSize(pts_on_iface, lc_inner)

#     gmsh.model.mesh.generate(3)

#     msh, ct, ft = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
#     gmsh.finalize()

#     dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
#     ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

#     dx_c = dx(1)   
#     dx_n = dx(2)   
#     ds_c = ds(11)  
#     ds_n = ds(12)  

#     return msh, ct, ft, dx_c, dx_n, ds_c, ds_n

def build_two_region_ball_mesh(R_outer: float = 1.0,
                               R_inner: float = 0.4,
                               lc_outer: float = 0.5,
                               lc_inner: float = 0.1):
    
    gmsh.initialize()
    gmsh.model.add("cell_with_nucleus_3d")

    v_outer = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, R_outer)
    v_inner = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, R_inner)

    out, _ = gmsh.model.occ.fragment([(3, v_outer)], [(3, v_inner)])
    gmsh.model.occ.synchronize()

    volumes = [ent for ent in out if ent[0] == 3]

    def bbox_radius(vol_tag):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(3, vol_tag)
        return max(abs(xmax), abs(ymax), abs(zmax))

    v_small, v_large = sorted([v[1] for v in volumes], key=bbox_radius)
    inner_vol = v_small
    outer_vol = v_large

    gmsh.model.addPhysicalGroup(3, [outer_vol], tag=1)  # Omega_c
    gmsh.model.setPhysicalName(3, 1, "Omega_c")
    gmsh.model.addPhysicalGroup(3, [inner_vol], tag=2)  # Omega_n
    gmsh.model.setPhysicalName(3, 2, "Omega_n")

    bnd_outer = gmsh.model.getBoundary([(3, outer_vol)], oriented=False, recursive=False)
    bnd_inner = gmsh.model.getBoundary([(3, inner_vol)], oriented=False, recursive=False)

    surf_outer = [s[1] for s in bnd_outer if s[0] == 2]
    surf_inner = [s[1] for s in bnd_inner if s[0] == 2]

    def surface_radius(surf_tag):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, surf_tag)
        return max(abs(xmax), abs(ymax), abs(zmax))

    outer_surface = max(surf_outer, key=surface_radius)
    interface_surface = max(surf_inner, key=surface_radius)

    gmsh.model.addPhysicalGroup(2, [outer_surface], tag=11)  # Gamma_c
    gmsh.model.setPhysicalName(2, 11, "Gamma_c")
    gmsh.model.addPhysicalGroup(2, [interface_surface], tag=12)  # Gamma_n
    gmsh.model.setPhysicalName(2, 12, "Gamma_n")

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc_outer)
    pts_on_iface = gmsh.model.getBoundary([(2, interface_surface)], oriented=False, recursive=True)
    pts_on_iface = [p for p in pts_on_iface if p[0] == 0]
    if pts_on_iface:
        gmsh.model.mesh.setSize(pts_on_iface, lc_inner)

    gmsh.model.mesh.generate(3)

    msh, ct, ft = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()

    dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    
    dx_c = dx(1)   
    dx_n = dx(2)   
    ds_c = ds(11)  
    ds_n = ds(12)  

    return msh, ct, ft, dx_c, dx_n, ds_c, ds_n
def build_spherical_shell_mesh(
    R_outer: float = 1.0,
    R_inner: float = 0.4,
    lc_outer: float = 0.5,
    lc_inner: float = 0.1,
):
    """
    3D mesh сферического слоя: { R_inner < r < R_outer }.

    Physical groups:
      - Volume (Omega_shell): tag=1
      - Outer boundary (r=R_outer): tag=11
      - Inner boundary (r=R_inner): tag=12

    Returns:
      msh, ct, ft, dx_shell, ds_outer, ds_inner
    """
    import gmsh
    import numpy as np
    from dolfinx import io
    from mpi4py import MPI
    import ufl

    gmsh.initialize()
    gmsh.model.add("spherical_shell_3d")

    v_outer = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, R_outer)
    v_inner = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, R_inner)

    # shell = outer \ inner
    shell_dimtags, _ = gmsh.model.occ.cut([(3, v_outer)], [(3, v_inner)],
                                          removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()

    shell_vol = [e[1] for e in shell_dimtags if e[0] == 3][0]
    gmsh.model.addPhysicalGroup(3, [shell_vol], tag=1)

    # collect all boundary surface patches
    bnd = gmsh.model.getBoundary([(3, shell_vol)], oriented=False, recursive=False)
    surf_tags = [s[1] for s in bnd if s[0] == 2]

    def surf_r(s):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, s)
        return max(abs(xmax), abs(ymax), abs(zmax))

    radii = np.array([surf_r(s) for s in surf_tags], dtype=float)
    rmin, rmax = radii.min(), radii.max()

    # group patches by radius (inner ~ rmin, outer ~ rmax)
    tol = 1e-8 + 1e-3 * max(R_outer, 1.0)
    inner_surfaces = [s for s, r in zip(surf_tags, radii) if abs(r - rmin) < tol]
    outer_surfaces = [s for s, r in zip(surf_tags, radii) if abs(r - rmax) < tol]

    gmsh.model.addPhysicalGroup(2, outer_surfaces, tag=11)
    gmsh.model.addPhysicalGroup(2, inner_surfaces, tag=12)

    # mesh sizes
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc_outer)
    pts_on_inner = gmsh.model.getBoundary([(2, inner_surfaces[0])], oriented=False, recursive=True)
    pts_on_inner = [p for p in pts_on_inner if p[0] == 0]
    if pts_on_inner:
        gmsh.model.mesh.setSize(pts_on_inner, lc_inner)

    gmsh.model.mesh.generate(3)

    msh, ct, ft = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()

    dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

    dx_shell = dx(1)
    ds_outer = ds(11)
    ds_inner = ds(12)

    return msh, ct, ft, dx_shell, ds_outer, ds_inner

def build_ball_mesh(
    R: float = 1.0,
    lc: float = 0.2,
):
    """
    3D mesh шара: { r < R }.

    Physical groups:
      - Volume (Omega): tag=1
      - Outer boundary (Gamma, r=R): tag=11

    Returns:
      msh, ct, ft, dx_ball, ds_outer
    """
    import gmsh
    import numpy as np
    from mpi4py import MPI
    from dolfinx import io
    import ufl

    gmsh.initialize()
    gmsh.model.add("ball_3d")

    v = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, R)
    gmsh.model.occ.synchronize()

    # volume tag
    gmsh.model.addPhysicalGroup(3, [v], tag=1)
    gmsh.model.setPhysicalName(3, 1, "Omega")

    # all boundary surface patches of the sphere -> tag=11
    bnd = gmsh.model.getBoundary([(3, v)], oriented=False, recursive=False)
    surf_tags = [s[1] for s in bnd if s[0] == 2]
    gmsh.model.addPhysicalGroup(2, surf_tags, tag=11)
    gmsh.model.setPhysicalName(2, 11, "Gamma_outer")

    # mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    gmsh.model.mesh.generate(3)

    msh, ct, ft = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()

    dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

    dx_ball = dx(1)
    ds_outer = ds(11)

    return msh, ct, ft, dx_ball, ds_outer
def mesh_update():
    pass


def build_disk_mesh(
    R: float = 1.0,
    lc: float = 0.05,
):
    import gmsh
    from mpi4py import MPI
    from dolfinx import io
    import ufl

    gmsh.initialize()
    gmsh.model.add("disk_2d")

    # круг → поверхность (2D!)
    disk = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, R, R)
    gmsh.model.occ.synchronize()

    # область
    gmsh.model.addPhysicalGroup(2, [disk], tag=1)

    # граница (окружность)
    boundary = gmsh.model.getBoundary([(2, disk)], oriented=False)
    curve_tags = [c[1] for c in boundary if c[0] == 1]

    gmsh.model.addPhysicalGroup(1, curve_tags, tag=11)

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
    gmsh.model.mesh.generate(2)

    msh, ct, ft = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()

    dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

    return msh, ct, ft, dx(1), ds(11)