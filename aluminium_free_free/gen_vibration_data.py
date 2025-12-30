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
def pick_measure_dof_near_point(V: fem.FunctionSpace, x0: np.ndarray, atol: float,
                                component: int = 2) -> int:
    """
    FRF를 CSV로 뽑을 때 '관심점 근처'의 DOF 하나를 대표로 고른다.
    (노드 정확히 일치가 아니라도 atol로 가장 가까운 것 찾는 느낌)
    """
    bs = V.dofmap.index_map_bs
    assert bs == 3

    def near_point(x):
        return (np.isclose(x[0], x0[0], atol=atol) &
                np.isclose(x[1], x0[1], atol=atol) &
                np.isclose(x[2], x0[2], atol=atol))

    cand = fem.locate_dofs_geometrical(V, near_point)
    cand = [int(d) for d in cand if (int(d) % bs) == component]
    if len(cand) == 0:
        # fallback: 그냥 가장 가까운 노드를 찾는 건 dolfinx API로 바로는 애매해서,
        # atol을 늘리는 걸 권장. 여기서는 안전하게 에러.
        raise RuntimeError("No measurement DOF found near x0. Increase atol.")
    return cand[0]

def build_patch_force_vector(V: fem.FunctionSpace, x0: np.ndarray, patch_half: np.ndarray,
                            F_total: float, component: int = 2) -> tuple[PETSc.Vec, np.ndarray]:
    """
    노드에 정확히 안 걸려도 되는 'patch 분포하중' 방식.
    x0 중심, (±patch_half) 박스 영역 내 DOF들에 force를 분배.
    - component: 0=x, 1=y, 2=z
    - F_total: 전체 힘 (N)
    반환: (f_vec, selected_dofs)
    """
    bs = V.dofmap.index_map_bs
    assert bs == 3, "Vector P1 expected (bs=3)."

    ndofs = V.dofmap.index_map.size_global * bs
    comm = V.mesh.comm

    # patch 안에 들어오는 dof 찾기 (geometrical)
    x_min = x0 - patch_half
    x_max = x0 + patch_half

    def in_patch(x):
        return ((x[0] >= x_min[0]) & (x[0] <= x_max[0]) &
                (x[1] >= x_min[1]) & (x[1] <= x_max[1]) &
                (x[2] >= x_min[2]) & (x[2] <= x_max[2]))

    dofs_patch_all = fem.locate_dofs_geometrical(V, in_patch)

    # 해당 component(z 등)에 해당하는 DOF만
    dofs_patch = np.array([int(d) for d in dofs_patch_all if (int(d) % bs) == component], dtype=np.int32)

    # MPI 환경에서 전체 개수 합산
    n_local = np.array([len(dofs_patch)], dtype=np.int64)
    n_global = np.array([0], dtype=np.int64)
    comm.Allreduce(n_local, n_global, op=MPI.SUM)

    if n_global[0] == 0:
        raise RuntimeError(
            f"No DOF in patch. Enlarge patch_half or move x0.\n"
            f"x0={x0}, patch_half={patch_half}"
        )

    # 각 DOF에 분배할 힘 (전체 합이 F_total이 되도록)
    f_each = F_total / float(n_global[0])

    f = PETSc.Vec().createMPI(ndofs, comm=comm)
    f.set(0.0)

    # 소유 구간에 있는 DOF만 setValue (안전)
    r0, r1 = f.getOwnershipRange()
    for d in dofs_patch:
        if r0 <= d < r1:
            f.setValue(d, f_each)

    f.assemblyBegin()
    f.assemblyEnd()

    return f, dofs_patch


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
    zm0= 0
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


    dofs_mass = fem.locate_dofs_geometrical(V, in_mass_block)

    if len(dofs_mass) == 0:
        raise RuntimeError("Mass block DOFs not found")
    ndofs = len(dofs_mass)
    m_per_dof = m_add / ndofs

    K = fem.petsc.assemble_matrix(a_form, diagonal=1/62831)
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


    # ---- 모달 중첩 ----


    modes = []     # ← 이게 modes
    freqs = []     # 고유진동수도 같이 저장

    for i in range(evs):
        lam_i = eigensolver.getEigenpair(i, vr, vi)
        if lam_i.real <= 0:
            continue

        f_i = np.sqrt(lam_i.real) / (2.0 * np.pi)
        if f_i < 5:
            continue

        # Copy eigenvector into Function
        mode = fem.Function(V)
        mode.x.array[:] = vr.array[:]

        # Create PETSc Vec view (local)
        phi = PETSc.Vec().createWithArray(mode.x.array, comm=MPI.COMM_WORLD)

        # mass norm = sqrt(phi^T M phi)
        Mphi = M.createVecRight()
        M.mult(phi, Mphi)
        mnorm_sq = phi.dot(Mphi)
        mnorm = np.sqrt(np.abs(mnorm_sq))

        # Normalize (avoid division by 0)
        if mnorm < 1e-30:
            continue
        mode.x.array[:] = mode.x.array[:] / mnorm

        modes.append(mode)
        freqs.append(float(f_i))

    if len(modes) == 0:
        raise RuntimeError("No valid modes found for modal superposition")
    
    omega_r = 2*np.pi*np.array(freqs)   # rad/s

    omega = 2*np.pi*freq
    x0 = np.array([0.0, Ly / 2.0, 0.0], dtype=np.float64)
    hx, hy, hz = Lx / nx, Ly / ny, Lz / nz
    patch_half = np.array([2*hx, 2*hy, 2*hz])  # 충분히 작게

    f_vec, dofs_patch = build_patch_force_vector(
        V,
        x0=x0,
        patch_half=patch_half,
        F_total=a_base,
        component=2   # z 방향
    )
    modal_force = np.zeros(len(modes), dtype=np.complex128)

    for r, mode in enumerate(modes):
        phi = PETSc.Vec().createWithArray(mode.x.array, comm=MPI.COMM_WORLD)
        modal_force[r] = phi.dot(f_vec)   # ← ★ 이게 가진 효과





    u_resp = np.zeros(ndofs, dtype=np.complex128)

    for r, mode in enumerate(modes):
        den = (omega_r[r]**2 - omega**2) + 2j*zeta*omega_r[r]*omega
        u_resp += (mode.x.array * modal_force[r]) / den


    coords = domain.geometry.x


    u_node = u_resp.reshape((-1, 3))  
    df = pd.DataFrame({
        "node_id": np.arange(coords.shape[0]),
        "x": coords[:,0],
        "y": coords[:,1],
        "z": coords[:,2],
        "ux": np.real(u_node[:,0]),
        "uy": np.real(u_node[:,1]),
        "uz": np.real(u_node[:,2]),
        "ux_abs": np.abs(u_node[:,0]),
        "uy_abs": np.abs(u_node[:,1]),
        "uz_abs": np.abs(u_node[:,2]),})
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
    if 'eigensolver' in locals():
        eigensolver.destroy()
    if 'st' in locals():
        st.destroy()
    if 'K' in locals():
        K.destroy()
    if 'M' in locals():
        M.destroy()


if __name__=="__main__":
    total_samples=600
    for i in range(total_samples):
        print(f"Generating sample {i+1}/{total_samples}...")
        while True:
            try:
                gen_vibration_data(i+400)
                break
            except RuntimeError as e:
                print(f"Error generating sample {i}: {e}. Retrying...")
    print("All samples generated.")