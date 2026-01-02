import ufl
from mpi4py import MPI
from dolfinx import io, mesh
import gmsh
def build_two_region_ball_mesh(R_outer: float = 1.0,
                               R_inner: float = 0.4,
                               lc_outer: float = 0.12,
                               lc_inner: float = 0.08):
    
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
