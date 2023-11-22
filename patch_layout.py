import numpy as np
import open3d


def compute_smoothed_vertices(vertices, triangles):

  vertices_deepcopy = np.copy(vertices)

  for vertex_index in range(len(vertices)):
      
    # Find triangles connected to the current vertex
    # a triangle is connected if it contains the index of the vertex
    triangles_index = np.where(triangles == vertex_index)[0]
    connected_triangles = triangles[triangles_index]

    # Calculate the centroids of each connected triangle
    centroids = []
    for triangle in connected_triangles:
      centroid = np.mean(vertices[triangle], axis=0)
      centroids.append(centroid)

    # Calculate the smoothed vertex by average of the centroids
    smoothed_vertex = np.mean(centroids, axis=0)

    # Append the new vertex to the list
    vertices_deepcopy[vertex_index] = smoothed_vertex


  print(vertices_deepcopy)
  return np.asarray(vertices_deepcopy)

# Load mesh
file_path = "cube.obj"

mesh = open3d.io.read_triangle_mesh(file_path)

# Ensure the mesh has vertex normals
mesh.compute_vertex_normals()

vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

smoothed_vertices = compute_smoothed_vertices(vertices, triangles)
mesh.vertices = open3d.utility.Vector3dVector(smoothed_vertices)
mesh.compute_vertex_normals()

smoothed_normals = np.asarray(mesh.vertex_normals)

# Create a LineSet for visualizing normals
lines = []
for i in range(len(mesh.vertices)):
    start_point = mesh.vertices[i]
    end_point = start_point + 0.1 * smoothed_normals[i]  # Adjust the scaling factor as needed
    lines.append([start_point, end_point])

line_set = open3d.geometry.LineSet()
line_set.points = open3d.utility.Vector3dVector(np.vstack(lines))
line_set.lines = open3d.utility.Vector2iVector(np.arange(len(lines)).reshape(-1, 2))

# Visualize the mesh and normals
open3d.visualization.draw_geometries([mesh, line_set])

#setNeighbor()

#SVDFit3Fast



'''SmoothingMax();
MakeNormals(normal);
  
setNeighbor();
SVDFit3Fast();
OriginalCoordinate();
if(ridge==1)
  setRidgeRavine();
'''

