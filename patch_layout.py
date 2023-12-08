import numpy as np
import open3d
from sklearn.linear_model import ElasticNet, Lasso, Ridge
import numpy as np


connected_triangles_vertices = []
def compute_smoothed_vertices(vertices, triangles):

  vertices_deepcopy = np.copy(vertices)

  for vertex_index in range(len(vertices)):
      
    # Find triangles connected to the current vertex
    triangles_index = np.where(triangles == vertex_index)[0]
    connected_triangles = triangles[triangles_index]
    connected_triangles_vertices.append(connected_triangles)

    # Calculate the centroids of each connected triangle
    centroids = []
    for triangle in connected_triangles:
      centroid = np.mean(vertices[triangle], axis=0)
      centroids.append(centroid)

    # Calculate the smoothed vertex by average of the centroids
    smoothed_vertex = np.mean(centroids, axis=0)

    # Append the new vertex to the list
    vertices_deepcopy[vertex_index] = smoothed_vertex

  return np.asarray(vertices_deepcopy)

# Load mesh
file_path = "male.obj"

mesh = open3d.io.read_triangle_mesh(file_path)

# Ensure the mesh has vertex normals
mesh.compute_vertex_normals()
normals = np.asarray(mesh.vertex_normals)

vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

smoothed_vertices = compute_smoothed_vertices(vertices, triangles)
mesh.vertices = open3d.utility.Vector3dVector(smoothed_vertices)

mesh.compute_vertex_normals()
smoothed_normals = np.asarray(mesh.vertex_normals)

# Compute the neighbor vertices for each vertex until a length of max_neigh (like in a graph)
def setNeighbor(index_vertice, k):
  if k == 0:
    return []
  
  neighbors = []
  for vertices_temp in connected_triangles_vertices[index_vertice]:
    for vertex in vertices_temp:
      if vertex != index_vertice:
        neighbors.append(vertex)
        if k > 1:
          neighbors += setNeighbor(vertex, k-1)

  return neighbors
  
max_neigh = 3
neighbors = []
for vertex_index in range(len(vertices)):
  neighbors_current = setNeighbor(vertex_index, max_neigh)

  # remove duplicates
  neighbors_current = list(set(neighbors_current))

  # remove the current vertex
  if vertex_index in neighbors_current:
    neighbors_current.remove(vertex_index)

  # removing those vertices whose normals make obtuse angles with the normal of the current vertex
  neighbors_current_temp = []
  for neighbor in neighbors_current:
    if np.dot(smoothed_normals[vertex_index], smoothed_normals[neighbor]) > 0:
      neighbors_current_temp.append(neighbor)
    
  neighbors_current = neighbors_current_temp

  neighbors.append(neighbors_current)


def normalize_3d(point):
    norm = np.linalg.norm(point)
    if norm == 0:
        norm = 1
    point = point / norm

    return point

def set_virtual_tangents(normal, out_t1, out_t2):
    if abs(normal[0]) < abs(normal[1]):
        out_t1[0], out_t1[1], out_t1[2] = 0.0, -normal[2], normal[1]
    else:
        out_t1[0], out_t1[1], out_t1[2] = normal[2], 0.0, -normal[0]

    out_t1 = normalize_3d(out_t1)

    out_t2[0] = (normal[1] * out_t1[2]) - (normal[2] * out_t1[1])
    out_t2[1] = (normal[2] * out_t1[0]) - (normal[0] * out_t1[2])
    out_t2[2] = (normal[0] * out_t1[1]) - (normal[1] * out_t1[0])

    out_t2 = normalize_3d(out_t2)

    return out_t1, out_t2

def cross_vector(out, in1, in2):
    out[0] = (in1[1] * in2[2]) - (in2[1] * in1[2])
    out[1] = (in1[2] * in2[0]) - (in2[2] * in1[0])
    out[2] = (in1[0] * in2[1]) - (in2[0] * in1[1])
    return out

def jacobi(a, tol=1e-10, max_iter=50):
  n = len(a)
  v = np.identity(n)  # Inizializza la matrice ortogonale v come matrice identità
  nrot = 0

  for _ in range(max_iter):
      sm = np.sum(np.abs(a) - np.abs(np.diag(a)))  # Somma degli elementi fuori diagonale

      if sm < tol:
          return np.diag(a), v, nrot

      tresh = 0.2 * sm / (n * n) if _ < 4 else 0.0

      for ip in range(n - 1):
          for iq in range(ip + 1, n):
              g = 100.0 * np.abs(a[ip, iq])

              if _ > 4 and (np.abs(a[ip, ip]) + g) == np.abs(a[ip, ip]) and \
                            (np.abs(a[iq, iq]) + g) == np.abs(a[iq, iq]):
                  a[ip, iq] = 0.0
              elif np.abs(a[ip, iq]) > tresh:
                  h = a[iq, iq] - a[ip, ip]
                  if np.abs(h) + g == np.abs(h):
                      t = a[ip, iq] / h
                  else:
                      theta = 0.5 * h / a[ip, iq]
                      t = 1.0 / (np.abs(theta) + np.sqrt(1.0 + theta**2))
                      if theta < 0.0:
                          t = -t

                  c = 1.0 / np.sqrt(1 + t**2)
                  s = t * c
                  tau = s / (1.0 + c)
                  h = t * a[ip, iq]
                  a[ip, iq] = 0.0
                  a -= h * np.outer(v[:, ip], v[:, iq])
                  a[:, ip] += h * v[:, iq]
                  a[:, iq] -= h * v[:, ip]
                  v[:, ip] -= h * v[:, iq]
                  v[:, iq] += h * v[:, ip]
                  nrot += 1

  raise ValueError("Too many iterations in routine jacobi")

# computing first derivative of principal curvature w.r.t their direction
def getk1t1(a00,a10,ab3, ab5, ab4, ab6):
  return 6.0*(a00*(a00*a00*ab3+a10*a10*ab5)+a10*(a00*a00*ab4+a10*a10*ab6))


bc = np.zeros((10, 3))
def surface_curvature(vertices):

  k1 = []
  k2 = []
  ks1 = []
  ks2 = []
  t1 = []
  t2 = []
  for vertex_index in range(len(vertices)):
    bt1, bt2 = set_virtual_tangents(smoothed_normals[vertex_index], np.zeros(3), np.zeros(3))

    A = []
    dkk = []
    for neighbor in neighbors[vertex_index]:

      bc[0] = vertices[neighbor] - vertices[vertex_index]
      dz = np.dot(bc[0], smoothed_normals[vertex_index])
      dUU = np.dot(bc[0], bt1)
      dVV = np.dot(bc[0], bt2)
      UU = dUU*dUU
      UV = dUU*dVV
      VV = dVV*dVV
      
      #norma = 2 / (UU + VV)
      norma = 1

      a_temp = np.zeros(8)
      A.append(a_temp)
      dkk.append(0)

      a_temp = np.zeros(8)
      a_temp[0] = 0
      a_temp[1] = 0.5*UU
      a_temp[2] = UV
      a_temp[3] = 0.5*VV
      a_temp[4] = dUU*UU
      a_temp[5] = UU*dVV
      a_temp[6] = dUU*VV
      a_temp[7] = VV*dVV
      A.append(a_temp * norma)
      
      dkk.append(dz * norma)
      
      a_temp = np.zeros(8)
      a_temp[0] = 0
      a_temp[1] = dUU
      a_temp[2] = dVV
      a_temp[3] = 0.0
      a_temp[4] = 3.0*UU
      a_temp[5] = 2.0*UV
      a_temp[6] = VV
      a_temp[7] = 0.0
      A.append(a_temp * norma)


      dx = np.dot(smoothed_normals[neighbor], bt1)
      dz = np.dot(smoothed_normals[neighbor], smoothed_normals[vertex_index])
      dy = np.dot(smoothed_normals[neighbor], bt2)
      
      dkk.append(-dx/dz * norma)
      
      a_temp = np.zeros(8)
      a_temp[0] = 0.0
      a_temp[1] = 0.0
      a_temp[2] = dUU
      a_temp[3] = dVV
      a_temp[4] = 0.0
      a_temp[5] = UU
      a_temp[6] = 2.0*UV
      a_temp[7] = 3.0*VV
      A.append(a_temp * norma)

      dkk.append(-dy/dz * norma)

    A = np.asarray(A)

    # risoluzione del sistema lineare
    ab = np.linalg.lstsq(A, dkk, rcond=0.0001)[0]

    eia = np.zeros((3,3))
    eia[1][1] = ab[1]
    eia[1][2] = ab[2]
    eia[2][1] = ab[2]
    eia[2][2] = ab[3]

    eid, eiv = np.linalg.eig(eia)

    if(eid[1]<eid[2]):
      k1.append(eid[2])
      k2.append(eid[1])

      t2_temp = np.zeros(3)
      t2_temp[0] = (bt1[0])*eiv[1][1] + (bt2[0])*eiv[2][1]
      t2_temp[1] = (bt1[1])*eiv[1][1] + (bt2[1])*eiv[2][1]
      t2_temp[2] = (bt1[2])*eiv[1][1] + (bt2[2])*eiv[2][1]
      t2.append(t2_temp)

      t1_temp = np.zeros(3)
      t1_temp[0] = (bt1[0])*eiv[1][2] + (bt2[0])*eiv[2][2]
      t1_temp[1] = (bt1[1])*eiv[1][2] + (bt2[1])*eiv[2][2]
      t1_temp[2] = (bt1[2])*eiv[1][2] + (bt2[2])*eiv[2][2]
      t1.append(t1_temp)
      
      t1x = eiv[1][2]
      t1y = eiv[2][2]
      t2x = eiv[1][1]
      t2y = eiv[2][1]
    else:
      k1.append(eid[1])
      k2.append(eid[2])
            
      t1_temp = np.zeros(3)
      t1_temp[0] = (bt1[0])*eiv[1][1] + (bt2[0])*eiv[2][1]
      t1_temp[1] = (bt1[1])*eiv[1][1] + (bt2[1])*eiv[2][1]
      t1_temp[2] = (bt1[2])*eiv[1][1] + (bt2[2])*eiv[2][1]
      t1.append(t1_temp)

      t2_temp = np.zeros(3)
      t2_temp[0] = (bt1[0])*eiv[1][2] + (bt2[0])*eiv[2][2]
      t2_temp[1] = (bt1[1])*eiv[1][2] + (bt2[1])*eiv[2][2]
      t2_temp[2] = (bt1[2])*eiv[1][2] + (bt2[2])*eiv[2][2]
      t2.append(t2_temp)

      t2x = eiv[1][2]
      t2y = eiv[2][2]
      t1x = eiv[1][1]
      t1y = eiv[2][1] 
    
    t1[vertex_index] = normalize_3d(t1[vertex_index])
    t2[vertex_index] = normalize_3d(t2[vertex_index])
    
    ks1.append(getk1t1(t1x,t1y,ab[4], ab[6], ab[5], ab[7]))
    ks2.append(getk1t1(t2x,t2y,ab[4], ab[6], ab[6], ab[7]))

  return k1, k2, ks1, ks2, t1, t2

k1, k2, ks1, ks2, t1, t2 = surface_curvature(vertices)

# load k1, k2, ks1, ks2, t1, t2

#k1 = np.loadtxt("/Users/matteobalice/Desktop/retopology/CCode/k1.txt")
#k2 = np.loadtxt("/Users/matteobalice/Desktop/retopology/CCode/k2.txt")
#ks1 = np.loadtxt("/Users/matteobalice/Desktop/retopology/CCode/ks1.txt")
#ks2 = np.loadtxt("/Users/matteobalice/Desktop/retopology/CCode/ks2.txt")
#t1 = np.loadtxt("/Users/matteobalice/Desktop/retopology/CCode/t1.txt")
#t2 = np.loadtxt("/Users/matteobalice/Desktop/retopology/CCode/t2.txt")

#print first 3 ks1 and ks2
print(ks1[:5])
print(ks2[:5])
edges_list = []
for i in range(len(vertices) * 2):
    edges_list.append([])
      

def setInter(alpha, beta, dv1, dv2):
  out = np.zeros(3)
  out[0] = (alpha * dv2[0] + beta * dv1[0]) / (alpha + beta)
  out[1] = (alpha * dv2[1] + beta * dv1[1]) / (alpha + beta)
  out[2] = (alpha * dv2[2] + beta * dv1[2]) / (alpha + beta)
  return out

ptail = []
def AppendP(ID,dx,dy,dz,ov1,ov2,dval1,dval2):
  now = np.zeros(8)
  now[0] = ID
  now[1] = dx
  now[2] = dy
  now[3] = dz
  now[4] = ov1
  now[5] = ov2
  now[6] = dval1
  now[7] = dval2
  
  ptail.append(now)
  
  return ID


int_E = 0
edges = []
def AppendE1(dv1,dv2):
  global int_E
  id_now = int_E
  now = np.zeros(3)
  now[0] = id_now
  now[1] = dv1
  now[2] = dv2
  
  edges.append(now)

  edges_list[dv1].append([id_now, dv2])
  edges_list[dv2].append([id_now, dv1])
  int_E += 1

  return id_now


def AppendE2(dv1,dv2):
  global int_E
  now = np.zeros(3)
  now[0] = int_E
  now[1] = dv1
  now[2] = dv2

  edges.append(now)
  int_E = int_E + 1

def ridge_ravine(ks1, t1):

  global bc, triangles, vertices, edges_list, edges
  x = 0
  edge01 = 0
  edge12 = 0
  edge20 = 0

  ksf0 = ks1[triangles[i][0]]
  if (np.dot(t1[triangles[i][0]], t1[triangles[i][1]]) < 0.0):
    ksf1 = -ks1[triangles[i][1]]
    bc[5][0] = -t1[triangles[i][1]][0]
    bc[5][1] = -t1[triangles[i][1]][1]
    bc[5][2] = -t1[triangles[i][1]][2]
  else:
    ksf1 = ks1[triangles[i][1]]
    bc[5][0] = t1[triangles[i][1]][0]
    bc[5][1] = t1[triangles[i][1]][1]
    bc[5][2] = t1[triangles[i][1]][2]

  if (ksf0 * ksf1 <= 0.0):
    bc[3] = vertices[triangles[i][1]] - vertices[triangles[i][0]]
    bc[4] = vertices[triangles[i][0]] - vertices[triangles[i][1]]
    
    if (ksf0 * np.dot(bc[3], t1[triangles[i][0]]) < 0.0 or ksf1 * np.dot(bc[4], bc[5]) < 0.0):
      edge01 = 1

  ksf1 = ks1[triangles[i][1]]
  if (np.dot(t1[triangles[i][1]], t1[triangles[i][2]]) < 0.0):
    ksf2 = -ks1[triangles[i][2]]
    bc[5][0] = -t1[triangles[i][2]][0]
    bc[5][1] = -t1[triangles[i][2]][1]
    bc[5][2] = -t1[triangles[i][2]][2]
  else:
    ksf2 = ks1[triangles[i][2]]
    bc[5][0] = t1[triangles[i][2]][0]
    bc[5][1] = t1[triangles[i][2]][1]
    bc[5][2] = t1[triangles[i][2]][2]

  if (ksf1 * ksf2 <= 0.0):
    bc[3] = vertices[triangles[i][2]] - vertices[triangles[i][1]]
    bc[4] = vertices[triangles[i][1]] - vertices[triangles[i][2]]


    if (ksf1 * np.dot(bc[3], t1[triangles[i][1]]) < 0.0 or ksf2 * np.dot(bc[4], bc[5]) < 0.0):
      edge12 = 1

  ksf2 = ks1[triangles[i][2]]

  
  if (np.dot(t1[triangles[i][2]], t1[triangles[i][0]]) < 0.0):
    ksf0 = -ks1[triangles[i][0]]
    bc[5][0] = -t1[triangles[i][0]][0]
    bc[5][1] = -t1[triangles[i][0]][1]
    bc[5][2] = -t1[triangles[i][0]][2]
  else:
    ksf0 = ks1[triangles[i][0]]
    bc[5][0] = t1[triangles[i][0]][0]
    bc[5][1] = t1[triangles[i][0]][1]
    bc[5][2] = t1[triangles[i][0]][2]
  
  if (ksf2 * ksf0 <= 0.0):

    bc[3] = vertices[triangles[i][0]] - vertices[triangles[i][2]]
    bc[4] = vertices[triangles[i][2]] - vertices[triangles[i][0]]

    
    if (ksf2 * np.dot(bc[3], t1[triangles[i][2]]) < 0.0 or ksf0 * np.dot(bc[4], bc[5]) < 0.0):
      edge20 = 1
  

  if (edge01 + edge12 + edge20 >= 2):
    x = x+1
    if (edge01 == 1):

      checkID = triangles[i][1] in [edge[1] for edge in edges_list[triangles[i][0]]]
      if not checkID:
        alpha = abs(ksf0)
        beta = abs(ksf1)

        bc[0] = setInter(alpha, beta, vertices[triangles[i][0]], vertices[triangles[i][1]])

        checkID = AppendP(AppendE1(triangles[i][0], triangles[i][1]), bc[0][0], bc[0][1], bc[0][2], triangles[i][0], triangles[i][1], alpha, beta)


      if edge12 == 1:
        
        checkID2 = triangles[i][2] in [edge[1] for edge in edges_list[triangles[i][1]]]
        if not checkID2:
            alpha = abs(ksf1)
            beta = abs(ksf2)

            bc[0] = setInter(alpha, beta, vertices[triangles[i][1]], vertices[triangles[i][2]])
            checkID2 = AppendP(AppendE1(triangles[i][1], triangles[i][2]), bc[0][0], bc[0][1], bc[0][2], triangles[i][1], triangles[i][2], alpha, beta)
      else:
        checkID2 = triangles[i][0] in [edge[1] for edge in edges_list[triangles[i][2]]]
        if not checkID2:
            alpha = abs(ksf2)
            beta = abs(ksf0)
            bc[0] = setInter(alpha, beta, vertices[triangles[i][2]], vertices[triangles[i][0]])
            checkID2 = AppendP(AppendE1(triangles[i][2], triangles[i][0]), bc[0][0], bc[0][1], bc[0][2], triangles[i][2], triangles[i][0], alpha, beta)
    else:
      checkID = triangles[i][2] in [edge[1] for edge in edges_list[triangles[i][1]]]
      if (not checkID):
        alpha = abs(ksf1)
        beta = abs(ksf2)
        bc[0] = setInter(alpha, beta, vertices[triangles[i][1]], vertices[triangles[i][2]])
        checkID = AppendP(AppendE1(triangles[i][1], triangles[i][2]), bc[0][0], bc[0][1], bc[0][2], triangles[i][1], triangles[i][2], alpha, beta)
      else:
        #get the edge id
        for edge in edges_list[triangles[i][1]]:
          if edge[1] == triangles[i][2]:
            checkID = edge[0]
            break
      
      checkID2 = triangles[i][0] in [edge[1] for edge in edges_list[triangles[i][2]]]
      if (not checkID2):
        alpha = abs(ksf2)
        beta = abs(ksf0)
        bc[0] = setInter(alpha, beta, vertices[triangles[i][2]], vertices[triangles[i][0]])
        checkID2 = AppendP(AppendE1(triangles[i][2], triangles[i][0]), bc[0][0], bc[0][1], bc[0][2], triangles[i][2], triangles[i][0], alpha, beta)
      else:
        #get the edge id
        for edge in edges_list[triangles[i][2]]:
          if edge[1] == triangles[i][0]:
            checkID2 = edge[0]
            break

    #AppendE2(checkID, checkID2)
  return x

bc = np.zeros((10, 3))

for i in range(len(triangles)):
    
    # Ridge: convex crest
    if k1[triangles[i][0]] > abs(k2[triangles[i][0]]) and k1[triangles[i][1]] > abs(k2[triangles[i][1]]) and k1[triangles[i][2]] > abs(k2[triangles[i][2]]):
      ridge_ravine(ks1, t1)

    # Ravine: concave crest
    if ((-abs(k1[triangles[i][0]])) > (k2[triangles[i][0]]) and (-abs(k1[triangles[i][1]])) > (k2[triangles[i][1]]) and (-abs(k1[triangles[i][2]])) > (k2[triangles[i][2]])):
      ridge_ravine(ks2, t2)
    


edges_print = []
vertices_print = []
dict_change_id = {}
for i in range(len(edges)):
    
  if(int(edges[i][1]) not  in dict_change_id):
    vertices_print.append(vertices[int(edges[i][1])])
    dict_change_id[edges[i][1]] = len(vertices_print) - 1
  
  if(int(edges[i][2]) not in dict_change_id):
    vertices_print.append(vertices[int(edges[i][2])])
    dict_change_id[edges[i][2]] = len(vertices_print) - 1

  edges_print.append([dict_change_id[int(edges[i][1])], dict_change_id[int(edges[i][2])]])

#write in a file:
# first line: number of vertices
# second line: number of edges

path = "output.txt"

file = open(path, "w")
file.write(str(len(vertices_print)) + "\n")
file.write(str(len(edges_print)) + "\n")

#write coordinates of vertices
for vertex in vertices_print:
  file.write(str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]) + "\n")

#write edges
for edge in edges_print:
  file.write(str(edge[0]) + " " + str(edge[1]) + "\n")

file.close()

# Creiamo un LineSet usando Open3D
lineset = open3d.geometry.LineSet()

# Aggiungiamo i punti al LineSet
lineset.points = open3d.utility.Vector3dVector(vertices_print)

# Aggiungiamo gli edges al LineSet
lineset.lines = open3d.utility.Vector2iVector(edges_print)

# Visualizziamo il LineSet
open3d.visualization.draw_geometries([lineset])