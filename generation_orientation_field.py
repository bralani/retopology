import numpy as np
from collections import defaultdict


def leggi_obj(file_path):
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



def generate_orientation_field(vertices_triangled, vertices_ground_truth, faces_ground_truth):

    # Definisci il vertice e le facce
    faces_coords = [[vertices_ground_truth[i-1] for i in face] for face in faces_ground_truth]
    # se la faccia è un tris, sostituisci i valori con infinito
    for face in faces_coords:
        if len(face) < 4:
            face.append([float('inf'), float('inf'), float('inf')])

    baricentri = np.array([np.mean(face, axis=0) for face in faces_coords])

    '''
    # Calcola la distanza tra il vertice e tutti i vertici della ground truth
    distanze = np.linalg.norm(np.array(vertices_ground_truth)[:, np.newaxis, :] - np.array(vertices_triangled)[np.newaxis, :, :], axis=2)

    # per ciascun vertice della ground truth, preleva le facce a cui appartiene e sommaci la distanza
    sum_faces = defaultdict(int)
    for i, distanza in enumerate(distanze):
        faces = vertices_to_faces[i + 1]
        for face in faces:
            sum_faces[face] += distanza

    for i, face in enumerate(faces_coords):
        if len(face) < 4:
            sum_faces[i] = float('inf')'''
    
    # Calcola la distanza tra il vertice e tutti i baricentri delle facce della ground truth
    distanze = np.linalg.norm(baricentri[:, np.newaxis, :] - np.array(vertices_triangled)[np.newaxis, :, :], axis=2)

    # Trova l'indice della faccia più vicina
    indice_faccia_piu_vicina = np.argmin(distanze, axis=0)

    return indice_faccia_piu_vicina


def proiezione_su_faccia(vertice, faccia):
    # Calcola il vettore normale alla faccia
    v0 = np.array(faccia[0])
    v1 = np.array(faccia[1])
    v2 = np.array(faccia[2])
    normale = np.cross(v1 - v0, v2 - v0)

    # Scegli uno dei punti sulla faccia come punto di riferimento
    punto_di_riferimento = v0

    # Calcola il vettore dal vertice al punto di riferimento sulla faccia
    v_vertice = np.array(vertice)
    vettore_vertice_riferimento = punto_di_riferimento - v_vertice

    # Calcola la distanza lungo il vettore normale dalla faccia al vertice
    distanza_normale = np.dot(vettore_vertice_riferimento, normale) / np.linalg.norm(normale)

    # Calcola la posizione della proiezione sulla faccia
    proiezione = vertice + (distanza_normale * normale)

    return proiezione

def distanza_vertice_a_segmento(vertice, punto1, punto2):
    punto1 = np.array(punto1)
    punto2 = np.array(punto2)
    # Calcola il vettore dal primo punto del segmento al vertice
    v_vertice_punto1 = vertice - punto1

    # Calcola il vettore direzione del segmento
    v_direzione = punto2 - punto1

    # Calcola la lunghezza del segmento
    lunghezza_segmento = np.linalg.norm(v_direzione)

    # Normalizza il vettore direzione del segmento
    v_direzione_normalizzato = v_direzione / lunghezza_segmento

    # Calcola la proiezione del vertice sul segmento
    proiezione = np.dot(v_vertice_punto1, v_direzione_normalizzato) / lunghezza_segmento

    # Gestisci i casi in cui la proiezione si trova al di fuori del segmento
    if proiezione < 0:
        punto_prossimo = punto1
    elif proiezione > 1:
        punto_prossimo = punto2
    else:
        punto_prossimo = punto1 + proiezione * v_direzione_normalizzato

    # Calcola la distanza tra il vertice e il punto più vicino sul segmento
    distanza = np.linalg.norm(vertice - punto_prossimo)

    return distanza

def generate_output(url_triangle, url_quads):

    vertices_ground_truth, faces = leggi_obj(url_quads)

    vertices_to_faces = {}
    for i, face in enumerate(faces):
        for vertex in face:
            if vertex in vertices_to_faces:
                vertices_to_faces[vertex].append(i)
            else:
                vertices_to_faces[vertex] = [i]


    vertices_triangle, _ = leggi_obj(url_triangle)

    orientation_fields = []

    indice_faccia = generate_orientation_field(vertices_triangle, vertices_ground_truth, faces)

    for idx_vertice, _ in enumerate(vertices_triangle):
        idx_faccia = indice_faccia[idx_vertice]
        faccia_coords = [vertices_ground_truth[i-1] for i in faces[idx_faccia]]

        p_primo = proiezione_su_faccia(vertices_triangle[idx_vertice], [faccia_coords[0], faccia_coords[1], faccia_coords[2]])

        d1 = distanza_vertice_a_segmento(p_primo, faccia_coords[0], faccia_coords[1])
        d2 = distanza_vertice_a_segmento(p_primo, faccia_coords[1], faccia_coords[2])
        d3 = distanza_vertice_a_segmento(p_primo, faccia_coords[2], faccia_coords[3])
        d4 = distanza_vertice_a_segmento(p_primo, faccia_coords[3], faccia_coords[0])

        # calculate the frame field directions u and v
        faccia_coords = np.array(faccia_coords)
        u = d3/(d1+d3) * (faccia_coords[1] - faccia_coords[0]) + d1/(d1+d3) * (faccia_coords[2] - faccia_coords[3])
        v = d4/(d2+d4) * (faccia_coords[2] - faccia_coords[1]) + d2/(d2+d4) * (faccia_coords[3] - faccia_coords[0])

        # rendi unitari i vettori u e v
        #u_normalized = u / np.linalg.norm(u)
        #v_normalized = v / np.linalg.norm(v)

        orientation_fields.append([u, v])

    return np.array(orientation_fields)




