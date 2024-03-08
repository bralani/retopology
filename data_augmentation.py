import trimesh
import numpy as np
import networkx as nx
import pyvista
import pyacvd


def leggi_file_output(file_path):
    vertici = []
    lati = []

    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.split()

            if not tokens:
                continue

            # Estrai vertici
            if tokens[0] == 'v':
                vertice = list(map(float, tokens[1:]))
                vertici.append(vertice)

            # Estrai lati
            elif tokens[0] == 'l':
                faccia = [int(vertex.split('/')[0]) for vertex in tokens[1:]]
                lati.append(faccia)

    return vertici, lati

def data_augmentation(url):

  print("Augmenting " + url + " ...")
  # take the name of the file
  file_name = url.split('/')[-1].split('.')[0]

  vertices_output, edges_output = leggi_file_output('dataset/original_dataset/output/' + file_name + '_output.obj')
  vertices_output.insert(0, [0, 0, 0])

  mesh = pyvista.read(url)

  subdivisions = [1, 2, 3]
  clusters = [10000, 20000, 30000, 40000]
  idx = 1
  for subdivide in subdivisions:
      for cluster in clusters:
        remeshing(mesh, vertices_output, edges_output, file_name, subdivide, cluster, idx)
        idx += 1


def remeshing(mesh, vertices_output, edges_output, file_name, subdivide, cluster, idx):
  clus = pyacvd.Clustering(mesh)
  clus.subdivide(subdivide)
  clus.cluster(cluster)
  
  # remesh
  remesh = clus.create_mesh()

  vertices = remesh.points
  triangles = remesh.faces


  # create a graph from the edges and vertices
  graph_output = nx.Graph()
  graph_output.add_nodes_from(range(len(vertices_output)))
  graph_output.add_edges_from(edges_output)

  # construct faces with the right format
  faces = []
  i = 0
  while i < len(triangles):
      n = triangles[i]
      faces.append(triangles[i+1:i+n+1])
      i += n+1

  faces = np.array(faces).tolist()

      
  mesh = trimesh.Trimesh(vertices=vertices,
                        faces=faces)



  # edges without duplication
  edges = mesh.edges_unique

  # the actual length of each unique edge
  length = mesh.edges_unique_length

  # create the graph with edge attributes for length
  g = nx.Graph()
  for edge, L in zip(edges, length):
      g.add_edge(*edge, length=L)


  # find the connected components of the graph
  components_output = list(nx.connected_components(graph_output))

  # remove all the components that has less than 100 vertices
  components_output = [component for component in components_output if len(component) > 2]

  cloud = trimesh.points.PointCloud(vertices_output)
  scene = trimesh.Scene([mesh,cloud])
  scene.show()



  map_vertex = {}
  vertex_output = []
  edges_output = []

  for component in components_output:
      visited = set()
      new_vertex = list(component)[0]
      visited.add(new_vertex)

      closest_point = mesh.nearest.vertex([vertices_output[new_vertex]])
      closest_point = first_closest_point = closest_point[1][0]

      if closest_point not in map_vertex:
          vertex_output.append(closest_point)
          map_vertex[closest_point] = len(vertex_output) - 1

      # get new vertex
      old_vertex = closest_point
      neighbors = list(graph_output.neighbors(new_vertex))
      for neighbor in neighbors:
          if neighbor not in visited and neighbor in list(component):
              new_vertex = neighbor
              break

      while len(visited) < len(list(component)):
          visited.add(new_vertex)

          old_closest_point = closest_point
          closest_point = mesh.nearest.vertex([vertices_output[new_vertex]])
          closest_point = closest_point[1][0]
          if closest_point not in map_vertex:
              vertex_output.append(closest_point)
              map_vertex[closest_point] = len(vertex_output) - 1

          # take the shortest path between the two points
          path = nx.shortest_path(g,
                                  source=old_closest_point,
                                  target=closest_point,
                                  weight='length')
                  
          # add the edges of the shortest path to the list
          for i in range(len(path)-1):
              if path[i] not in map_vertex:
                  vertex_output.append(path[i])
                  map_vertex[path[i]] = len(vertex_output) - 1

              if path[i+1] not in map_vertex:
                  vertex_output.append(path[i+1])
                  map_vertex[path[i+1]] = len(vertex_output) - 1
                  
              edges_output.append([map_vertex[path[i]], map_vertex[path[i+1]]])

          # get new vertex
          old_vertex = new_vertex
          neighbors = list(graph_output.neighbors(new_vertex))
          for neighbor in neighbors:
              if neighbor not in visited and neighbor in list(component):
                  new_vertex = neighbor
                  break
              
          if old_vertex == new_vertex:
              break

        # check if the last point is connected to the first point
      neighbors = list(graph_output.neighbors(new_vertex))
      for neighbor in neighbors:
        if neighbor == list(component)[0]:

            # take the shortest path between the last and the first point
            path = nx.shortest_path(g,
                                    source=closest_point,
                                    target=first_closest_point,
                                    weight='length')
                    
            # add the edges of the shortest path to the list
            for i in range(len(path)-1):
                if path[i] not in map_vertex:
                    vertex_output.append(path[i])
                    map_vertex[path[i]] = len(vertex_output) - 1

                if path[i+1] not in map_vertex:
                    vertex_output.append(path[i+1])
                    map_vertex[path[i+1]] = len(vertex_output) - 1
                    
                edges_output.append([map_vertex[path[i]], map_vertex[path[i+1]]])

            break

  # save the output
  trimesh.exchange.export.export_mesh(mesh, 'dataset/augmented_dataset/input/' + file_name + str(idx) +'.obj')
  file = open("dataset/augmented_dataset/output/" + file_name + str(idx) + "_output.obj", "w")

  for vertex in vertex_output:
      file.write("v " + str(mesh.vertices.base[vertex][0]) + " " + str(mesh.vertices.base[vertex][1]) + " " + str(mesh.vertices.base[vertex][2]) + "\n")

  for edge in edges_output:
      file.write("l " + str(edge[0] + 1) + " " + str(edge[1] + 1) + "\n")

  file.close()
  file = open("dataset/augmented_dataset/output/" + file_name + str(idx) + "_output.txt", "w")

  for vertex in vertex_output:
      file.write(str(vertex) + " ")

  file.close()