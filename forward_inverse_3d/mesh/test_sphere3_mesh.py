import gmsh
import sys

# 1. 初始化 Gmsh
gmsh.initialize()

# 可选：设置网格的通用尺寸参数
lc = 0.1 # Characteristic length (控制网格细度)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc * 2)

# 2. 定义几何实体 (使用 OpenCASCADE 内核，推荐用于复杂实体)
gmsh.model.add("my_complex_geometry")
# 假设我们想创建一个外部大球体和两个内部的椭球体


# 2. 定义几何实体 (只做一个半球)
R_outer = 1.5
full_sphere = gmsh.model.occ.addSphere(0, 0, 0, R_outer)
box = gmsh.model.occ.addBox(-R_outer, -R_outer, 0, R_outer*2, R_outer*2, R_outer)
left_cavity = gmsh.model.occ.addSphere(R_outer/2, 0, 0, R_outer/2)
right_cavity = gmsh.model.occ.addSphere(-R_outer/2, 0, 0, R_outer/2)
gmsh.model.occ.cut([(3, full_sphere)], [(3, box)], tag=7, removeObject=True, removeTool=True)
gmsh.model.occ.cut([(3, 7)], [(3, left_cavity), (3, right_cavity)], removeTool=False)
gmsh.model.occ.fragment([(3, 7)], [(3, left_cavity), (3, right_cavity)])
gmsh.model.occ.synchronize()

# 3. 生成网格
gmsh.model.addPhysicalGroup(3, [7], 1)
gmsh.model.addPhysicalGroup(3, [left_cavity], 2) # 内部腔体作为另一个物理组
gmsh.model.addPhysicalGroup(3, [right_cavity], 3) # 内部腔体作为另一个物理组
gmsh.option.setNumber("Mesh.Algorithm", 6) # 6 = Delaunay
gmsh.model.mesh.generate(3)

# 4. 保存网格
gmsh.write("hemisphere.msh")

# 5. 可选：运行 Gmsh GUI 查看结果
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

# 6. 终止 Gmsh
gmsh.finalize()