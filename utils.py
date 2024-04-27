import numpy as np
import math, os


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




