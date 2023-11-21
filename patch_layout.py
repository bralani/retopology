import numpy as np
import open3d

# Load mesh
ply_path = "male.ply"
mesh = open3d.io.read_triangle_mesh(ply_path)

#vertices = np.asarray(mesh.vertices)
#faces = np.asarray(mesh.triangles)

# Rendering mesh
mesh.compute_vertex_normals()
open3d.visualization.draw_geometries([mesh])
