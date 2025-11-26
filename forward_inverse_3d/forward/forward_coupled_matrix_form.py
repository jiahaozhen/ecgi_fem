from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Function
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.mesh import create_submesh
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from mpi4py import MPI
import h5py
from utils.simulate_tools import build_Mi, build_M, ischemia_condition
from utils.function_tools import assign_function
from utils.ventricular_segmentation_tools import distinguish_epi_endo

def build_forward_matrix_coupled(mesh_file,
                                 sigma_i=0.4, sigma_e=0.8, sigma_t=0.8,
                                 multi_flag=True, gdim=3,
                                 center_ischemia=np.array([80.4, 19.7, -15.0]),
                                 radius_ischemia=30,
                                 ischemia_epi_endo=[-1]):
    """
    Build transfer matrix M_transfer: (n_targets, ndofs_V2)
    Solves A u = -R[:, i] for each basis i of V2, then samples u at closest torso points.
    """
    # ------------- meshes & spaces -------------
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim

    # submesh of ventricles: cell marker 2 assumed (same as original)
    subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    V1 = functionspace(domain, ("Lagrange", 1))
    V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))

    # epicardium/mid/endocardium markers
    epi_endo_marker = distinguish_epi_endo(mesh_file, gdim=gdim)
    marker_function = Function(V2)
    # assign_function expects values per mesh node or dof depending on implementation; keep your usage
    assign_function(marker_function, np.arange(len(subdomain_ventricle.geometry.x)), epi_endo_marker)

    condition = ischemia_condition(u_ischemia=1.0, u_healthy=0.0,
                                   center=center_ischemia,
                                   r=radius_ischemia,
                                   marker_function=marker_function,
                                   ischemia_epi_endo=ischemia_epi_endo)

    Mi = build_Mi(subdomain_ventricle, condition, sigma_i=sigma_i,
                  scar=False,
                  ischemia=False)
    M = build_M(domain, cell_markers, multi_flag=multi_flag, condition=condition,
                sigma_i=sigma_i, sigma_e=sigma_e, sigma_t=sigma_t,
                scar=False,
                ischemia=False)

    # ------------- assemble A and R -------------
    u1 = TrialFunction(V1)
    u2 = TrialFunction(V2)
    v1 = TestFunction(V1)

    dx1 = Measure("dx", domain=domain)
    a_element = dot(grad(v1), dot(M, grad(u1))) * dx1
    bilinear_form_a = form(a_element)
    A = assemble_matrix(bilinear_form_a)
    A.assemble()

    dx2 = Measure("dx", domain=subdomain_ventricle)
    r_element = dot(grad(v1), dot(Mi, grad(u2))) * dx2
    # map entities from subdomain -> domain
    entity_map = {domain._cpp_object: ventricle_to_torso}
    bilinear_form_r = form(r_element, entity_maps=entity_map)
    R = assemble_matrix(bilinear_form_r)
    R.assemble()

    # Guarantee we have PETSc.Mat objects (assemble_matrix may already return PETSc Mat)
    try:
        A_mat = A.mat()
    except Exception:
        A_mat = A
    try:
        R_mat = R.mat()
    except Exception:
        R_mat = R

    from petsc4py import PETSc
    nullspace = PETSc.NullSpace().create(constant=True, comm=MPI.COMM_WORLD)
    A_mat.setNullSpace(nullspace)

    # ------------- target points and mapping -------------
    geom = h5py.File(r'forward_inverse_3d/data/geom_ecgsim.mat', 'r')
    # geom_thorax/pts shape can be (3,300) or (300,3) depending on mat file; normalize
    target_pts = np.array(geom['geom_thorax']['pts'])
    if target_pts.ndim == 2 and target_pts.shape[0] == 3 and target_pts.shape[1] != 3:
        target_pts = target_pts.T
    target_pts = np.asarray(target_pts, dtype=float)  # (ntargets, 3)

    # V1 tabulate coords: ensure shape (ndofs_V1, gdim)
    f_space_pts = V1.tabulate_dof_coordinates()
    f_space_pts = np.asarray(f_space_pts, dtype=float)
    # many dolfinx returns shape (ndofs, gdim) already; if it's (gdim, ndofs) transpose
    if f_space_pts.ndim == 2 and f_space_pts.shape[0] == 3 and f_space_pts.shape[1] != 3:
        f_space_pts = f_space_pts.T

    # compute nearest torso dof for each target electrode
    from scipy.spatial.distance import cdist
    dists = cdist(target_pts, f_space_pts)  # shape: (ntargets, ndofs_V1)
    closest_indices = np.argmin(dists, axis=1)  # (ntargets,)

    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    try:
        ai, aj, av = A_mat.getValuesCSR()  # ai, aj 以 petsc 的 CSR 表示 (ia, ja, a)
    except Exception as e:
        raise RuntimeError("无法从 A_mat 获取 CSR。请确保 A 是稀疏 AIJ 矩阵。错误: " + str(e))

    # PETSc 返回的 ai 是 row pointers 长度 nrows+1
    nrows = A_mat.getSize()[0]
    A_csr = sp.csr_matrix((av, aj, ai), shape=(nrows, nrows))

    # ---- 将 A 转为适合直接因式分解的格式（CSC）并做一次 LU 分解 ----
    # 注意：splu 要求 csc 格式且是方阵
    A_csc = A_csr.tocsc()

    # 如果 A 非对称或奇异需额外处理；你原来设置了 nullspace，确保 rhs 已正交到 nullspace
    # LU 因式分解（可能占用内存，视矩阵稀疏结构而定）
    lu = spla.splu(A_csc)  # 若内存不足或矩阵巨大，这一步会失败 -> 改用并行 direct solver (见注)

    # ---- 获取 R 的全部列（或分块读取以节省内存） ----
    # 理想：一次性把 R 写成稠密数组 cols_count = ndofs_V2
    ndofs_V1 = int(A_mat.getSize()[0])
    ndofs_V2 = int(R_mat.getSize()[1])
    ntargets = len(closest_indices)

    # helper: 以块形式读取 R 矩阵列，避免内存峰值（chunk_size 可调）
    chunk_size = 200  # 根据内存与自由基数调节
    M_transfer = np.zeros((ntargets, ndofs_V2), dtype=np.float64)

    # PETSc Mat 提取列的通用方式是 getValuesCSR 或直接把整个 R 变为稀疏矩阵
    # 尝试直接拿到 R 的 CSR（如果 R_mat 是 AIJ）
    try:
        r_ai, r_aj, r_av = R_mat.getValuesCSR()
        R_csr = sp.csr_matrix((r_av, r_aj, r_ai), shape=(ndofs_V1, ndofs_V2))
        use_R_csr = True
    except Exception:
        use_R_csr = False

    # 分块处理
    for start in range(0, ndofs_V2, chunk_size):
        end = min(start + chunk_size, ndofs_V2)
        k = end - start

        # 提取 R 的这块列作为稠密矩阵 (ndofs_V1, k)
        if use_R_csr:
            R_block = R_csr[:, start:end].toarray()  # 可能大，但比一次性取全部小
        else:
            # fallback: 按列从 PETSc 读（较慢），但分块比逐列好
            R_block = np.zeros((ndofs_V1, k), dtype=np.float64)
            for j, col_idx in enumerate(range(start, end)):
                vec = R_mat.getColumnVector(col_idx)  # PETSc Vec
                R_block[:, j] = vec.getArray()

        # RHS = -R_block
        RHS_block = -R_block

        # 若有 nullspace，需要把 RHS 正交到 nullspace（你原来用 nullspace.remove(rhs)）
        # 对稠密块：将每列正交化（nullspace 是 constant 向量，因此投影很简单）
        # 假设 nullspace 是常数向量: 投影到常数子空间并移除
        # (更通用的做法需要使用 PETSc.NullSpace 的 apply 方法；这里做常量投影示例)
        # compute mean per column and subtract mean*1-vector
        col_means = RHS_block.mean(axis=0)
        RHS_block = RHS_block - col_means[np.newaxis, :]

        # 使用 LU 因式分解求解 U_block = A^{-1} * RHS_block
        # splu 只支持解单个 RHS 向量或多列矩阵 via .solve
        U_block = np.empty_like(RHS_block)
        for j in range(k):
            U_block[:, j] = lu.solve(RHS_block[:, j])

        # 现在在 NumPy 层对所有解向量做采样
        M_transfer[:, start:end] = U_block[closest_indices, :]  # shape: (ntargets, k)

        # optional progress
        # print(f"[build_forward_matrix_coupled] processed columns {start}..{end-1} / {ndofs_V2}", flush=True)

    # 最终返回 M_transfer
    return M_transfer


def forward_tmp(mesh_file, v_data,
                sigma_i=0.4, sigma_e=0.8, sigma_t=0.8,
                multi_flag=True, gdim=3, allow_cache=True):
    """
    Use the transfer matrix A (300 x N_heart) to compute BSP (300 x T) for given v_data.
    v_data may be:
      - 1D array of length N_nodes  -> treated as single timepoint
      - 2D array (N_nodes, T)
      - 2D array (T, N_nodes) -> transposed automatically
    """
    # 构建前向矩阵 (300 × Nheart)
    file_path = 'forward_inverse_3d/data/' + mesh_file.split('/')[-1].replace('.msh', '_forward_matrix_coupled.npz')
    try:
        if not allow_cache:
            raise FileNotFoundError
        data = np.load(file_path)
        A_transfer_matrix = data['A']
        print(f"Loaded precomputed forward matrix from {file_path}.", flush=True)
    except FileNotFoundError:
        print(f"Precomputed forward matrix not found at {file_path}, building anew...", flush=True)

        A_transfer_matrix = build_forward_matrix_coupled(
            mesh_file,
            sigma_i=sigma_i,
            sigma_e=sigma_e,
            sigma_t=sigma_t,
            multi_flag=multi_flag,
            gdim=gdim
        )

        if allow_cache:
            np.savez(file_path, A=A_transfer_matrix)
    
    A = np.asarray(A_transfer_matrix, dtype=float)  # (300, N_nodes)

    # make v into (N_nodes, T)
    v = np.asarray(v_data, dtype=float)

    if v.ndim == 1:
        # single timepoint vector
        if v.shape[0] == A.shape[1]:
            v = v.reshape(A.shape[1], 1)
        else:
            raise ValueError(f"v 的长度 {v.shape} 与 A 的列数 {A.shape[1]} 不匹配。")
    elif v.ndim == 2:
        # could be (N_nodes, T) or (T, N_nodes)
        if v.shape[0] == A.shape[1]:
            # (N_nodes, T) OK
            pass
        elif v.shape[1] == A.shape[1]:
            # (T, N_nodes) -> transpose
            v = v.T
        else:
            # ambiguous shape
            raise ValueError(f"无法识别 v 的形状 {v.shape}，应为 (N_nodes, T) 或 (T, N_nodes)，其中 N_nodes={A.shape[1]}.")
    else:
        raise ValueError("v_data 必须是一维或二维数组。")

    # final sanity check
    if A.shape[1] != v.shape[0]:
        raise ValueError(
            f"矩阵维度不匹配: A 是 {A.shape}, v 是 {v.shape}. "
            f"A 的列数必须等于 v 的节点数!"
        )

    # core multiplication: (300, N) x (N, T) -> (300, T)
    phi_body = A @ v

    # if single timepoint, return 1D array
    if phi_body.shape[1] == 1:
        return phi_body[:, 0]

    return phi_body


def compute_d_from_tmp(mesh_file, v_data,
                       sigma_i=0.4, sigma_e=0.8, sigma_t=0.8,
                       multi_flag=True, gdim=3,
                       allow_cache=False):
    d_data = forward_tmp(mesh_file, v_data,
                         sigma_i=sigma_i, sigma_e=sigma_e, sigma_t=sigma_t,
                         multi_flag=multi_flag, gdim=gdim, allow_cache=allow_cache)
    # your original function returns transposed
    return d_data.T