import numpy as np
import math, os
import trimesh
import numpy as np
import networkx as nx
import pyvista
import pyacvd
import random
import os


# read the obj file and return the vertices and faces
def read_obj(file_path):
    vertici = []
    facce = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertice = list(map(float, line.strip().split()[1:]))
                vertici.append(vertice)
            elif line.startswith('f '):
                face = line.strip().split()[1:]
                # Controllo se la faccia è un quad o un tris
                if len(face) == 4:
                    face = line.strip().split()[1:]
                faccia = [int(idx.split('/')[0]) for idx in face]
                facce.append(faccia)

    return vertici, facce


# return the index of the face closest to the vertex
def closest_face(vertices_triangled, vertices_ground_truth, faces_ground_truth):

    faces_coords = [[vertices_ground_truth[i-1] for i in face] for face in faces_ground_truth]

    for face in faces_coords:
        if len(face) < 4:
            face.append([float('inf'), float('inf'), float('inf')])

    baricentri = np.array([np.mean(face, axis=0) for face in faces_coords])
    
    distanze = np.linalg.norm(baricentri[:, np.newaxis, :] - np.array(vertices_triangled)[np.newaxis, :, :], axis=2)
    indice_faccia_piu_vicina = np.argmin(distanze, axis=0)

    return indice_faccia_piu_vicina


# project the vertex on the face
def project_vertex_face(vertice, faccia):
    # Calculate the normal vector of the face
    v0 = np.array(faccia[0])
    v1 = np.array(faccia[1])
    v2 = np.array(faccia[2])
    normale = np.cross(v1 - v0, v2 - v0)

    punto_di_riferimento = v0

    v_vertice = np.array(vertice)
    vettore_vertice_riferimento = punto_di_riferimento - v_vertice

    distanza_normale = np.dot(vettore_vertice_riferimento, normale) / np.linalg.norm(normale)

    proiezione = vertice + (distanza_normale * normale)

    return proiezione

# calculate the distance between a vertex and a segment
def distance_vertex_from_segment(vertice, punto1, punto2):
    punto1 = np.array(punto1)
    punto2 = np.array(punto2)

    v_vertice_punto1 = vertice - punto1
    v_direzione = punto2 - punto1

    lunghezza_segmento = np.linalg.norm(v_direzione)

    v_direzione_normalizzato = v_direzione / lunghezza_segmento

    proiezione = np.dot(v_vertice_punto1, v_direzione_normalizzato) / lunghezza_segmento

    if proiezione < 0:
        punto_prossimo = punto1
    elif proiezione > 1:
        punto_prossimo = punto2
    else:
        punto_prossimo = punto1 + proiezione * v_direzione_normalizzato

    distanza = np.linalg.norm(vertice - punto_prossimo)

    return distanza

# convert the cartesian coordinates to spherical coordinates
def cartesian_to_spherical(vet):

    # normalize the vector
    norm = np.linalg.norm(vet)
    vet = vet / norm

    x, y, z = vet

    theta = math.acos(z)
    phi = math.atan2(y, x)

    return theta, phi

def generate_orientations(file):
  file_name = os.path.basename(file)
  file_triangle = "./new_dataset/test/input/triangles/" + file_name

  orientation_fields = generate_output(file_triangle, file)
  orientation_fields_reshaped = orientation_fields.reshape(orientation_fields.shape[0], -1)

  # save the orientation fields in a txt file
  output_file = "./new_dataset/test/output/" + file_name
  np.savetxt(output_file, orientation_fields_reshaped)


  print("Orientation fields saved in: ", output_file)


def generate_output(url_triangle, url_quads):

    vertices_ground_truth, faces = read_obj(url_quads)

    vertices_to_faces = {}
    for i, face in enumerate(faces):
        for vertex in face:
            if vertex in vertices_to_faces:
                vertices_to_faces[vertex].append(i)
            else:
                vertices_to_faces[vertex] = [i]


    vertices_triangle, _ = read_obj(url_triangle)

    orientation_fields = []

    indice_faccia = closest_face(vertices_triangle, vertices_ground_truth, faces)

    for idx_vertice, _ in enumerate(vertices_triangle):
        idx_faccia = indice_faccia[idx_vertice]
        faccia_coords = [vertices_ground_truth[i-1] for i in faces[idx_faccia]]

        # project the vertex on the face
        p_primo = project_vertex_face(vertices_triangle[idx_vertice], [faccia_coords[0], faccia_coords[1], faccia_coords[2]])

        d1 = distance_vertex_from_segment(p_primo, faccia_coords[0], faccia_coords[1])
        d2 = distance_vertex_from_segment(p_primo, faccia_coords[1], faccia_coords[2])
        d3 = distance_vertex_from_segment(p_primo, faccia_coords[2], faccia_coords[3])
        d4 = distance_vertex_from_segment(p_primo, faccia_coords[3], faccia_coords[0])

        # calculate the frame field directions u and v
        faccia_coords = np.array(faccia_coords)
        u = d3/(d1+d3) * (faccia_coords[1] - faccia_coords[0]) + d1/(d1+d3) * (faccia_coords[2] - faccia_coords[3])
        v = d4/(d2+d4) * (faccia_coords[2] - faccia_coords[1]) + d2/(d2+d4) * (faccia_coords[3] - faccia_coords[0])

        u = cartesian_to_spherical(u)
        v = cartesian_to_spherical(v)

        orientation_fields.append([u, v])

    return np.array(orientation_fields)

def read_file_output(file_path):
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

def faces_to_edges(faces):
    edges = []
    for face in faces:
        for i in range(len(face)):
            edge = [face[i], face[(i+1)%len(face)]]
            edges.append(edge)
    return edges

def rotate(url):

    # Carica il file OBJ
    vertices, faces = read_obj(url)

    # Estrai i vertici dalla scena OBJ
    vertices = np.array(vertices)

    # Genera angoli casuali per le rotazioni su tutti e tre gli assi
    theta_x = np.random.uniform(0, 2 * np.pi)
    theta_y = np.random.uniform(0, 2 * np.pi)
    theta_z = np.random.uniform(0, 2 * np.pi)


    # Matrici di rotazione attorno agli assi
    rotation_matrix_x = np.array([[1, 0, 0],
                                [0, np.cos(theta_x), -np.sin(theta_x)],
                                [0, np.sin(theta_x), np.cos(theta_x)]])

    rotation_matrix_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                                [0, 1, 0],
                                [-np.sin(theta_y), 0, np.cos(theta_y)]])

    rotation_matrix_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                                [np.sin(theta_z), np.cos(theta_z), 0],
                                [0, 0, 1]])

    rotated_vertices = vertices
    rotated_vertices = np.dot(rotated_vertices, rotation_matrix_x.T)
    rotated_vertices = np.dot(rotated_vertices, rotation_matrix_y.T)
    rotated_vertices = np.dot(rotated_vertices, rotation_matrix_z.T)

    rotated_vertices = scale(rotated_vertices)


    # read the file 
    file = open(url, "r")
    lines = file.readlines()
    file.close()

    # write the new file
    file = open(url, "w")
    idx = 0
    for line in lines:
        if line.startswith('v '):
            vertex = rotated_vertices[idx]
            file.write("v " + str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]) + "\n")
            idx += 1
        else:
            file.write(line)


def scale(vertices):

    # Genera fattori di scala uguali per tutti e tre gli assi
    scale_factors = np.random.uniform(0.5, 2.0)

    # Applica la scala ai vertici
    scaled_vertices = vertices * scale_factors

    return scaled_vertices

def rotate_scale(url):
    print("Augmenting " + url + " ...")

    rotate(url)

def generate_trimesh(url):

    print("Preprocessing " + url + " ...")
    # take the name of the file
    file_name = os.path.basename(url).split('.')[0]
    #file_name = url.split('/')[-1].split('.')[0]

    vertices_output, faces_output = read_obj(url)
    edges_output = faces_to_edges(faces_output)
    vertices_output.insert(0, [0, 0, 0])

    mesh = pyvista.read(url)

    subdivisions = [random.randint(1,3)]
    clusters = [random.randrange(10000, 20001)]
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

    # save the output
    trimesh.exchange.export.export_mesh(mesh, 'dataset/train/input/triangles/' + file_name + '.obj')
