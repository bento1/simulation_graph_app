from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc
from dolfinx.io import XDMFFile
from dolfinx import mesh, fem
from dolfinx.fem import Constant,Function
import ufl
from ufl import TrialFunction, TestFunction, sym, grad, tr, Identity, inner, dx
from ufl import VectorElement
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import isExistFolder, makeFolder
import random
import json

# Convenience functions

def get_von_mises(sx, sy, sz, txy, tyz, tzx):
    return np.sqrt(
        0.5 * (
            (sx - sy)**2 +
            (sy - sz)**2 +
            (sz - sx)**2
        )
        + 3.0 * (txy**2 + tyz**2 + tzx**2)
    )   

name="sample"
root_path=r'./data'
def gen_vibration_data(sample_num):
    path=f"{root_path}/{name}_{sample_num:04d}"
    if not isExistFolder(path):
        makeFolder(path)

    params={}
    Lx =random.uniform(1, 10)
    Ly = random.uniform(0.1, 10)
    Lz =random.uniform(0.1, 10)
    nx= random.randint(10, 50)  # a <= x <= b
    ny= random.randint(5, 20) 
    nz= random.randint(5, 20) 
    xm0  = random.uniform(Lx/nx, Lx//2)
    xm1= xm0+Lx/nx*2
    ym0= random.uniform(0, Ly-Ly/ny)
    ym1 = ym0+Ly/ny*2
    zm0= random.uniform(0, Lz-Lz/nz)
    zm1 = zm0+Lz/nz*2
    E = 6.9e10
    nu = 0.33
    rho = 2700
    m_add = random.uniform(1, 100)
    freq = random.uniform(80, 300)
    a_base = random.uniform(1, 20)
    zeta = 0.005

    params['Lx']=Lx
    params['Ly']=Ly
    params['Lz']=Lz
    params['nx']=nx
    params['ny']=ny
    params['nz']=nz
    params['xm0']=xm0
    params['xm1']=xm1
    params['ym0']=ym0
    params['ym1']=ym1
    params['zm0']=zm0
    params['zm1']=zm1
    params['E']=E
    params['nu']=nu
    params['rho']=rho
    params['m_add']=m_add
    params['freq']=freq
    params['a_base']=a_base
    params['zeta']=zeta

    domain = mesh.create_box(
        MPI.COMM_WORLD,
        np.array([[0.0, 0.0, 0.0],
                [Lx,  Ly,  Lz]]),
        [nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron
    )

    Ve = VectorElement("Lagrange", domain.ufl_cell(), 1)
    V = fem.FunctionSpace(domain, Ve)
    u = TrialFunction(V)
    v = TestFunction(V)
    mu = Constant(domain, E/2./(1+nu))
    lambda_ = Constant(domain, E*nu/(1+nu)/(1-2*nu))
    def eps(u):
        return ufl.sym(ufl.grad(u))
    def sigma(u):
        dim = u.ufl_shape[0]
        return lambda_ * ufl.nabla_div(u) * ufl.Identity(dim) + 2 * mu * eps(u)



    a = inner(sigma(u), eps(v)) * dx
    a_form = fem.form(a)
    # Mass
    m = rho * inner(u, v) * dx
    m_form = fem.form(m)
    def in_mass_block(x):
        return ((x[0] >= xm0) & (x[0] <= xm1) &
                (x[1] >= ym0) & (x[1] <= ym1) &
                (x[2] >= zm0) & (x[2] <= zm1))
    def clamp_region(x):
        return (
            np.isclose(x[0], Lx/2, atol=Lx/nx*2) &  # 중앙 x=0.5 ± 1cm
            np.isclose(x[2], 0, atol=Lz/nz*2)  # 바닥
        )

    dofs_mass = fem.locate_dofs_geometrical(V, in_mass_block)
    dofs_clamp = fem.locate_dofs_geometrical(V, clamp_region)
    zero = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
    bc = fem.dirichletbc(zero, dofs_clamp, V)
    bcs = [bc]
    if len(dofs_mass) == 0:
        raise RuntimeError("Mass block DOFs not found")
    ndofs = len(dofs_mass)
    m_per_dof = m_add / ndofs

    K = fem.petsc.assemble_matrix(a_form, bcs=bcs, diagonal=1/62831)
    K.assemble()

    M = fem.petsc.assemble_matrix(m_form, diagonal=62831)
    M.assemble()

    for dof in dofs_mass:
        M.setValue(dof, dof, m_per_dof, addv=True)

    M.assemble()

    # Create and configure eigenvalue solver
    N_eig = 400
    eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
    eigensolver.setDimensions(N_eig)
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    st = SLEPc.ST().create(MPI.COMM_WORLD)
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(120)
    st.setFromOptions()
    eigensolver.setST(st)
    eigensolver.setOperators(K, M)
    eigensolver.setFromOptions()

    # Compute eigenvalue-eigenvector pairs
    eigensolver.solve()
    evs = eigensolver.getConverged()
    vr, vi = K.getVecs()
    u_output = Function(V)
    u_output.name = "Eigenvector"
    print( "Number of converged eigenpairs %d" % evs )


    ndofs = M.getSize()[0]
    bs = V.dofmap.index_map_bs

    # ---- 베이스 영향 벡터 b (z방향) ----
    b = PETSc.Vec().createMPI(ndofs, comm=M.comm)
    b.set(0.0)

    clamp_set = set(map(int, dofs_clamp))
    r0, r1 = b.getOwnershipRange()
    for dof in range(r0, r1):
        if (dof % bs) == 2 and (dof not in clamp_set):
            b.setValue(dof, 1.0)

    b.assemblyBegin()
    b.assemblyEnd()

    Mb = M.createVecRight()
    M.mult(b, Mb)

    # ---- 모달 중첩 ----
    u_resp = np.zeros(ndofs, dtype=np.complex128)

    modes = []     # ← 이게 modes
    freqs = []     # 고유진동수도 같이 저장

    for i in range(evs):
        l = eigensolver.getEigenpair(i, vr, vi)

        # 고유치 → 주파수
        if l.real <= 0:
            continue

        _freq = np.sqrt(l.real) / (2*np.pi)

        # 저주파(강체모드) 컷
        if _freq < 5.0:
            continue

        # PETSc Vec → dolfinx Function
        mode = Function(V)
        mode.x.array[:] = vr.array[:]  
        mode.name = f"mode_{i}"
        modes.append(mode)
        freqs.append(_freq)

    if len(modes) == 0:
        raise RuntimeError("No valid modes found for modal superposition")
    
    omega_r = 2*np.pi*np.array(freqs)   # rad/s

    omega = 2*np.pi*freq

    for r, mode in enumerate(modes):
        phi = PETSc.Vec().createWithArray(mode.x.array, comm=M.comm)
        Fr = - phi.dot(Mb) * a_base
        den = (omega_r[r]**2 - omega**2) + 2j*zeta*omega_r[r]*omega
        u_resp += (mode.x.array * Fr) / den


    u_resp[np.array(dofs_clamp, dtype=np.int32)] = 0.0
    
    u_real = fem.Function(V)
    u_real.name = f"u_real_{round(freq)}Hz"
    u_real.x.array[:] = np.real(u_resp)


    We = ufl.TensorElement("DG", domain.ufl_cell(), degree=0, shape=(3, 3))
    W = fem.FunctionSpace(domain, We)

    stress_cell = fem.Function(W)
    stress_expr = fem.Expression(sigma(u_real), W.element.interpolation_points())
    stress_cell.interpolate(stress_expr)
    Wn = fem.FunctionSpace(
        domain,
        ufl.TensorElement("Lagrange", domain.ufl_cell(), 1, shape=(3,3))
    )

    stress_node = fem.Function(Wn)
    stress_node.name = f"stress_{round(freq)}Hz"

    w = ufl.TrialFunction(Wn)
    v = ufl.TestFunction(Wn)

    a_proj = ufl.inner(w, v) * ufl.dx
    L_proj = ufl.inner(stress_cell, v) * ufl.dx

    problem = fem.petsc.LinearProblem(
        a_proj, L_proj,
        u=stress_node,
        petsc_options={"ksp_type": "cg", "pc_type": "jacobi"}
    )
    problem.solve()
    S = stress_node.x.array.reshape((-1, 9))

    sig_xx = S[:,0]
    sig_yy = S[:,4]
    sig_zz = S[:,8]

    tau_xy = S[:,1]
    tau_yz = S[:,5]
    tau_zx = S[:,6]
    coords = domain.geometry.x


    coords_nodes = domain.geometry.x
    u_vec = u_real.x.array.reshape((-1, 3))  # NOTE: P1 vector면 "node 수"와 맞는 경우가 많지만, 안전하게는 아래 3.2를 권장


    df = pd.DataFrame({
        "node_id": np.arange(coords_nodes.shape[0]),
        "x": coords[:,0],
        "y": coords[:,1],
        "z": coords[:,2],
        "sigma_xx": sig_xx,
        "sigma_yy": sig_yy,
        "sigma_zz": sig_zz,
        "tau_xy": tau_xy,
        "tau_yz": tau_yz,
        "tau_zx": tau_zx,
        "ux": u_vec[:,0],
        "uy": u_vec[:,1],
        "uz": u_vec[:,2],
    })
    df['von_mises'] = df.apply(lambda x:get_von_mises(x['sigma_xx'], x['sigma_yy'], x['sigma_zz'],
                                                    x['tau_xy'], x['tau_yz'], x['tau_zx']), axis=1)
    df.to_csv(f"{path}/nodal_stress_disp.csv", index=False)
    print("saved: nodal_stress_disp.csv")



    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    conn = domain.topology.connectivity(tdim, 0)

    rows = []
    for cell_id in range(conn.num_nodes):
        nodes = conn.links(cell_id)
        row = {"cell_id": cell_id}
        for i, n in enumerate(nodes):
            row[f"node_{i}"] = int(n)
        rows.append(row)

    df_cells = pd.DataFrame(rows)
    df_cells.to_csv(f"{path}/edge_infos.csv", index=False)
    print("saved: edge_infos.csv")

    with open(f"{path}/params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print("saved: params.txt")


if __name__=="__main__":
    total_samples=250
    for i in range(total_samples):
        print(f"Generating sample {i+1}/{total_samples}...")
        while True:
            try:
                gen_vibration_data(i+750)
                break
            except RuntimeError as e:
                print(f"Error generating sample {i}: {e}. Retrying...")
    print("All samples generated.")