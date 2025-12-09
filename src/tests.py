def mini_test_mesh(msh, ct, ft, dx_c, dx_n, ds_c, ds_n, R_outer=1.0, R_inner=0.4,
                   xdmf_path: str = "cell_with_nucleus_mesh.xdmf",
                   show_pyvista: bool = False,
                   save_screenshot: bool = True,
                   screenshot_path: str = "mesh_preview.png") -> None:
    """Run checks/preview for the two-region disk mesh."""
    import math
    import numpy as np
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import fem, io

    comm = msh.comm
    rank = comm.rank

    # --- basic diagnostics ---
    n_cells_local = msh.topology.index_map(msh.topology.dim).size_local
    n_facets_local = msh.topology.index_map(msh.topology.dim - 1).size_local

    def tag_counts(tags):
        uniq, cnt = np.unique(tags.values, return_counts=True)
        return dict(zip(uniq.tolist(), cnt.tolist()))

    cell_counts = tag_counts(ct)
    facet_counts = tag_counts(ft)

    if rank == 0:
        print("=== Mesh mini-test ===")
        print(f"dim = {msh.topology.dim}")
        print(f"cells(local) = {n_cells_local}")
        print(f"facets(local) = {n_facets_local}")
        print("cell tags:", sorted(cell_counts.items()))
        print("facet tags:", sorted(facet_counts.items()))

    # --- numerical integrals vs analytics ---
    one = fem.Constant(msh, PETSc.ScalarType(1.0))
    area_c = fem.assemble_scalar(fem.form(one * dx_c))
    area_n = fem.assemble_scalar(fem.form(one * dx_n))
    perim_c = fem.assemble_scalar(fem.form(one * ds_c))
    perim_n = fem.assemble_scalar(fem.form(one * ds_n))

    pi = math.pi
    area_c_ref = pi * (R_outer**2 - R_inner**2)
    area_n_ref = pi * (R_inner**2)
    perim_c_ref = 2 * pi * R_outer
    perim_n_ref = 2 * pi * R_inner

    rel = lambda a, b: abs(a - b) / max(1.0, abs(b))
    if rank == 0:
        print(f"area cytoplasm    = {area_c:.6f}  (ref {area_c_ref:.6f})  rel.err = {rel(area_c, area_c_ref):.3e}")
        print(f"area nucleus       = {area_n:.6f}  (ref {area_n_ref:.6f})  rel.err = {rel(area_n, area_n_ref):.3e}")
        print(f"perimeter outer    = {perim_c:.6f}  (ref {perim_c_ref:.6f})  rel.err = {rel(perim_c, perim_c_ref):.3e}")
        print(f"perimeter nucleus  = {perim_n:.6f}  (ref {perim_n_ref:.6f})  rel.err = {rel(perim_n, perim_n_ref):.3e}")

    # --- cell diameter stats (optional) ---
    try:
        from ufl import CellDiameter
        Vdg = fem.functionspace(msh, ("DG", 0))
        h_proj = fem.Function(Vdg)
        h_expr = fem.Expression(CellDiameter(msh), Vdg.element.interpolation_points())
        h_proj.interpolate(h_expr)
        h_vals = h_proj.x.array
        hs = comm.gather(h_vals.copy(), root=0)
        if rank == 0 and hs:
            import numpy as _np
            h_all = _np.concatenate(hs)
            if h_all.size:
                print(f"cell diameter h: min={h_all.min():.4e}, mean={h_all.mean():.4e}, max={h_all.max():.4e}")
    except Exception as e:
        if rank == 0:
            print("CellDiameter stats skipped:", e)

    # --- write XDMF with tags ---
    with io.XDMFFile(comm, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_meshtags(ct, msh.geometry)
        xdmf.write_meshtags(ft, msh.geometry)
    if rank == 0:
        print(f"→ XDMF saved to: {xdmf_path} (open in ParaView; color by 'ct' and 'ft')")

    # --- pyvista preview (optional) ---
    if show_pyvista:
        try:
            import pyvista as pv
            from dolfinx.plot import create_vtk_mesh
            if rank == 0:
                try:
                    pv.start_xvfb()
                except Exception:
                    pass
                grid_cells = pv.UnstructuredGrid(*create_vtk_mesh(msh, msh.topology.dim, ct))
                grid_cells.cell_data["region"] = ct.values.astype(np.int32)
                grid_facets = pv.UnstructuredGrid(*create_vtk_mesh(msh, msh.topology.dim - 1, ft))
                grid_facets.cell_data["facet_tag"] = ft.values.astype(np.int32)

                p = pv.Plotter()
                p.add_mesh(grid_cells, show_edges=True, scalars="region", nan_color="lightgray", opacity=0.35)
                p.add_mesh(grid_facets, scalars="facet_tag", line_width=3)
                p.add_axes()
                p.add_text("Omega_c=1 (annulus), Omega_n=2 (disk)\nGamma_c=11 (outer), Gamma_n=12 (interface)", font_size=10)
                if save_screenshot:
                    p.show(screenshot=screenshot_path, auto_close=False)
                    print(f"→ Screenshot saved: {screenshot_path}")
                    p.close()
                else:
                    p.show()
        except Exception as e:
            if rank == 0:
                print("PyVista preview skipped:", e)

    # --- ASCII fallback preview ---
    if rank == 0:
        try:
            import numpy as _np
            from dolfinx import geometry
            bb = geometry.BoundingBoxTree(msh, msh.topology.dim)
            nx = ny = 32
            xs = _np.linspace(-R_outer * 1.05, R_outer * 1.05, nx)
            ys = _np.linspace(-R_outer * 1.05, R_outer * 1.05, ny)
            canvas = []
            for y in ys[::-1]:
                row = []
                for x in xs:
                    p = _np.array([x, y, 0.0])[:msh.geometry.dim]
                    cells = geometry.compute_collisions(bb, p)
                    if len(cells) == 0:
                        row.append(" ")
                    else:
                        c = int(cells[0])
                        tag = ct.values[ct.indices == c]
                        row.append("." if tag.size == 0 else ("1" if tag[0] == 1 else ("2" if tag[0] == 2 else "?")))
                canvas.append("".join(row))
            print("ASCII preview (coarse):")
            print("\n".join(canvas))
        except Exception as e:
            print("ASCII preview skipped:", e)