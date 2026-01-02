# Test of the dual-mesh building by the function build_two_region_ball_mesh()
# u functions set up manually to be able to distinguish between 

from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from dolfinx import fem, io
from dolfinx.mesh import create_submesh
from src.mesh_initialization import build_two_region_ball_mesh

msh, ct, ft, dx_c, dx_n, ds_c, ds_n = build_two_region_ball_mesh()

ids = np.unique(ct.values)
print("Cell tags:", ids)  

outer_id = 1  
inner_id = 2  
print(f"outer_id (cyto) = {outer_id}, inner_id (nucleus) = {inner_id}")

tdim = msh.topology.dim
gdim = msh.geometry.dim


cells_inner = ct.indices[ct.values == inner_id]  
cells_outer = ct.indices[ct.values == outer_id]  

res_inner = create_submesh(msh, tdim, cells_inner)
res_outer = create_submesh(msh, tdim, cells_outer)

msh_inner = res_inner[0]  
msh_outer = res_outer[0]  


V_inner = fem.functionspace(msh_inner, ("Lagrange", 1, (gdim,)))
V_outer = fem.functionspace(msh_outer, ("Lagrange", 1, (gdim,)))

u_inner = fem.Function(V_inner, name="u_inner")   
u_outer = fem.Function(V_outer, name="u_outer")   

def u_inner_expr(x):
    return np.vstack((
        0.2 * x[0],
        0.2 * x[1],
        0.2 * x[2],
    ))

u_inner.interpolate(u_inner_expr)

def u_outer_expr(x):
    return np.vstack((
        -x[1],        # u_x = -y
        x[0],         # u_y =  x
        0.0 * x[2],   # u_z =  0
    ))

u_outer.interpolate(u_outer_expr)


with io.XDMFFile(MPI.COMM_WORLD, "data/nucleus_test.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh_inner)
    xdmf.write_function(u_inner)

with io.XDMFFile(MPI.COMM_WORLD, "data/cytoplasm_test.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh_outer)
    xdmf.write_function(u_outer)
