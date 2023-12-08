import open3d
import numpy as np
import networkx as nx
import bezier
import matplotlib.pyplot as plt
import sys

'''
 * ridges.txt/ravines.txt: 
 1st line is a number of convex/concave crest line points (Integer). 
 2nd line is a number of convex/concave crest line edges. 
 3rd line is a number of crest line connected components (Integer). 
 
 Let us notate V, E, and N for above numbers, respectively. 
 Starting from 4th line, there are V lines which include three Double and one Integer values in one 
 line as "%lf %lf %lf %d": x,y,z of crest line point and the connected component ID. 
 The line number of the file minus 4 represents a crest point ID. 

 Next, there are N lines which include three Double values in one line: 
 Ridgeness, Sphericalness and Cyclideness for each corresponded connected component ID 
 (ex. 1st line of this section corresponds ID 0 of the connected component, 
 2nd line of them represents ID 1 and so on). 

 Finally, there are E lines which include three Integer values: 
 pair of crest points ID (edge) and the triangle ID of original mesh which includes this edge 
 (if there is -1 of that triangle ID then that edge is a connecting edge, see the paper).
 '''

# Load mesh
file_path = "male.obj"

mesh = open3d.io.read_triangle_mesh(file_path)
mesh.compute_vertex_normals()

vertices_real = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

# legge il file ravines.txt
file = open("CCode/ravines.txt", "r")
lines = file.readlines()
file.close()

# 1 linea: numero di vertici
num_vertices = int(lines[0].split()[0])

# 2 linea: numero di edges
num_edges = int(lines[1].split()[0])

# 3 linea: numero di componenti connesse
num_connected_components = int(lines[2].split()[0])

# V lines: x,y,z of crest line point and the connected component ID
vertices = []
for i in range(3, 3 + num_vertices):
    line = lines[i].split()
    vertices.append([float(line[0]), float(line[1]), float(line[2])])

# E lines: pair of crest points ID (edge) and the triangle ID of original mesh which includes this edge
edges = []
for i in range(3 + num_vertices + num_connected_components, 3 + num_vertices + num_connected_components + num_edges):
    line = lines[i].split()
    edges.append([int(line[0]), int(line[1])])


# legge il file ridges.txt
file = open("CCode/ridges.txt", "r")
lines = file.readlines()
file.close()

# 1 linea: numero di vertici
num_vertices2 = int(lines[0].split()[0])

# 2 linea: numero di edges
num_edges2 = int(lines[1].split()[0])

# 3 linea: numero di componenti connesse
num_connected_components2 = int(lines[2].split()[0])

# V lines: x,y,z of crest line point and the connected component ID
for i in range(3, 3 + num_vertices2):
    line = lines[i].split()
    vertices.append([float(line[0]), float(line[1]), float(line[2])])

# E lines: pair of crest points ID (edge) and the triangle ID of original mesh which includes this edge
for i in range(3 + num_vertices2 + num_connected_components2, 3 + num_vertices2 + num_connected_components2 + num_edges2):
    line = lines[i].split()
    edges.append([int(line[0]) + num_vertices, int(line[1]) + num_vertices])

num_vertices = num_vertices + num_vertices2
num_edges = num_edges + num_edges2


edges = sorted(edges)

# legge il file output.txt
file2 = open("CCode/output.txt", "r")
lines = file2.readlines()
file2.close()

# 1 linea: numero di vertici
num_vertices2 = int(lines[0].split()[0])

# 2 linea: numero di facce
num_faces = int(lines[1].split()[0])

# V lines: k1 e k2 di ogni vertice
k = []
t_max = []
t_min = []
normals = []
for i in range(2, 2 + num_vertices2):
    line = lines[i].split()
    k.append([float(line[0]), float(line[1])])
    t_max.append([float(line[2]), float(line[3]), float(line[4])])
    t_min.append([float(line[5]), float(line[6]), float(line[7])])
    normals.append([float(line[8]), float(line[9]), float(line[10])])




def mappa_vertici(array_vertici_da_mappare, array_vertici_di_riferimento):
    mapping = {}
    mapping_inverse = {}
    idx = 0
    for vertice in array_vertici_da_mappare:
        
        distanze = np.linalg.norm(array_vertici_di_riferimento - vertice, axis=1)
        indice_vertice_piu_vicino = np.argmin(distanze)

        mapping[idx] = indice_vertice_piu_vicino

        # check se il vertice è già stato mappato
        if indice_vertice_piu_vicino in mapping_inverse:
            mapping_inverse[indice_vertice_piu_vicino].append(idx)
        else:
            mapping_inverse[indice_vertice_piu_vicino] = [idx]

        idx += 1
    return mapping, mapping_inverse

mapping, mapping_inverse = mappa_vertici(vertices, vertices_real)

def longest_simple_paths(G, source, target):
    longest_paths = []
    longest_path_length = 0
    for path in nx.all_simple_paths(G, source=source, target=target):
        if len(path) > longest_path_length:
            longest_path_length = len(path)
            longest_paths.clear()
            longest_paths.append(path)
        elif len(path) == longest_path_length:
            longest_paths.append(path)
    return longest_paths

def find_isolated_edges(nodes, edges):
    # convert each elements of edges in a tuple
    edges_tuple = [(edge[0], edge[1]) for edge in edges]

    # Creazione di un grafo diretto
    G = nx.Graph()

    # Aggiunta di nodi ed archi al grafo
    G.add_nodes_from(nodes)
    G.add_edges_from(edges_tuple)
    
    # Estrai le curve isolate
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    sub_graphs = [curve.edges for curve in S]

    for i in range(len(sub_graphs)):
        sub_graphs[i] = [[e[0], e[1]] for e in sub_graphs[i]]

    return sub_graphs


def filter_edges(isolated_edges):
    filtered_edges = []
    array_R = []

    # calculate the length of each curve
    for curve in isolated_edges:
        length = 0
        for i in range(len(curve) - 1):
            length += np.linalg.norm(np.array(vertices_real[curve[i][0]]) - np.array(vertices_real[curve[i][1]]))

        array_R.append(length)

    # ordina le curve in base alla lunghezza
    array_R, isolated_edges = zip(*sorted(zip(array_R, isolated_edges)))
    percentile = np.percentile(array_R,20)

    for i in range(len(array_R)):
        if array_R[i] > percentile:
            filtered_edges.append(isolated_edges[i])


    return filtered_edges


nodes = [mapping[i] for i in range(num_vertices)]

for idx in range(len(edges) - 1):
    edges[idx] = [mapping[edges[idx][0]], mapping[edges[idx][1]]]

edges = [coppia for coppia in edges if coppia[0] != coppia[1]]

connected_edges = find_isolated_edges(nodes, edges)
filtered_edges = filter_edges(connected_edges)


for curve in filtered_edges:
    # cerca l'elemento che compare una sola volta nella prima colonna
    # preleva la prima colonna
    first_column = np.array(curve)[:, 0].tolist()
    second_column = np.array(curve)[:, 1].tolist()

    #preleva la posizione dell'elemento che compare una sola volta nella prima colonna e mai nella seconda
    lista = [x for x in first_column if first_column.count(x) == 1 and x not in second_column]

    if len(lista)  > 0:
        idx = first_column.index(lista[0])

        # allora metti il vertice iniziale all'inizio della curva
        curve.insert(0, curve[idx])
        curve.pop(idx + 1)

    #cerca l'elemento che compare una sola volta nella seconda colonna
    
    lista = [x for x in second_column if second_column.count(x) == 1 and x not in first_column]
    if len(lista)  > 0:
        idx = second_column.index(lista[0])

        # allora metti il vertice finale alla fine della curva
        curve.append(curve[idx])
        curve.pop(idx)

'''
def crea_curva(nodo, map_edges):
    nuove_curve = []
    actual_curva = []
    actual_curva.append(nodo)
    while True:
        if nodo in map_edges:
            if len(map_edges[nodo]) == 1:
                next_node = map_edges[nodo][0]
                actual_curva.append(next_node)

                del map_edges[nodo]
                nodo = next_node
            elif len(map_edges[nodo]) > 1:
                
                for edge in map_edges[nodo]:
                    new_curves = crea_curva(edge, map_edges)
                    # aggiunge nodo all'inizio di ogni curva
                    for curve in new_curves:
                        curve.insert(0, nodo)
                        nuove_curve.append(curve)
                
                del map_edges[nodo]

                break
            else:
                break
        else:
            break
        
    nuove_curve.append(actual_curva)
    return nuove_curve
'''

# sistema filtered_edges in modo che ogni curva abbia all'inizio il vertice iniziale e alla fine il vertice finale
for i in range(len(filtered_edges)):
    curve = filtered_edges[i]

    #cerca l'elemento che compare una sola volta nella prima colonna
    # preleva la prima colonna
    first_column = np.array(curve)[:, 0].tolist()
    second_column = np.array(curve)[:, 1].tolist()

    #preleva la posizione dell'elemento che compare una sola volta nella prima colonna e mai nella seconda
    lista1 = [x for x in first_column if first_column.count(x) == 1 and x not in second_column]
    lista2 = [x for x in second_column if second_column.count(x) == 1 and x not in first_column]

    if len(lista1) == 0 and len(lista2) > 1:
        idx = second_column.index(lista2[0])

        curve.insert(0, [curve[idx][1], curve[idx][0]])
        curve.pop(idx + 1)
    elif len(lista1)  > 0:
        idx = first_column.index(lista1[-1])

        # allora metti il vertice iniziale all'inizio della curva
        curve.insert(0, curve[idx])
        curve.pop(idx + 1)
    
    if len(lista2) == 0 and len(lista1) > 1:
        idx = first_column.index(lista1[0])

        # allora metti il vertice iniziale all'inizio della curva
        curve.append([curve[idx][1], curve[idx][0]])
        curve.pop(idx)
    elif len(lista2)  > 0:
        idx = second_column.index(lista2[-1])

        # allora metti il vertice finale alla fine della curva
        curve.append(curve[idx])
        curve.pop(idx)


    filtered_edges[i] = curve


def longest_path(edges):
    
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Specifica i nodi di partenza e destinazione
    start_node = edges[0][0]
    end_node = edges[-1][1]

    # Ottieni tutti i percorsi semplici tra i nodi di partenza e destinazione
    all_paths = list(nx.all_simple_paths(G, source=start_node, target=end_node))

    # Trova il percorso più lungo
    longest_path = max(all_paths, key=len)

    return longest_path




# suddivide le curve in modo che ci siano tutti vertici con al massimo due archi uscenti
grafo = {}
filtered_edges_temp = []
for curve in filtered_edges:
    
    curve = longest_path(curve)
    curve_temp = []

    for i in range(len(curve) - 1):
        curve_temp.append([curve[i], curve[i + 1]])

    filtered_edges_temp.append(curve_temp)

filtered_edges = filtered_edges_temp
 

neighbors_vertex = []
def calculate_neighbors(triangles):
    for i in range(len(vertices_real)):
        neighbors_vertex.append([])

    for triangle in triangles:
        neighbors_vertex[triangle[0]].append(triangle[1])
        neighbors_vertex[triangle[0]].append(triangle[2])

        neighbors_vertex[triangle[1]].append(triangle[0])
        neighbors_vertex[triangle[1]].append(triangle[2])

        neighbors_vertex[triangle[2]].append(triangle[0])
        neighbors_vertex[triangle[2]].append(triangle[1])

    return neighbors_vertex
neighbors_vertex = calculate_neighbors(triangles)

def one_ring_neighborhood(vertex, n_edge):
    neighborhood = set()
    
    # Aggiungi i vicini del vertice
    neighborhood.update(neighbors_vertex[vertex])

    # Converti maps_curve in un set di tuple per evitare duplicati
    curve_set = set(tuple(sorted((el[0], el[1]))) for el in filtered_edges[n_edge])

    map1, map2 = zip(*curve_set)

    # Rimuovi duplicati da neighborhood e converti in lista
    neighborhood = list(set(neighborhood).difference(set(map1)).difference(set(map2)))

    return neighborhood


def calculate_cost(vertice, candidates, n_edge):
    candidates = np.array(candidates)
    
    k_temp = np.array(k[vertice]) - np.array(k)[candidates]
    k1 = k_temp[:, 0]
    k2 = k_temp[:, 1]
    t1 = np.array(t_max[vertice]) - np.array(t_max)[candidates]
    t2 = np.array(t_min[vertice]) - np.array(t_min)[candidates]

    normals_vertice = np.array(normals[vertice]).reshape((3,1))
    normals_candidates = np.array(normals)[candidates]

    prodotto_scalare = np.dot(normals_candidates, normals_vertice)

    # Calcola le norme dei vettori
    norma_a = np.linalg.norm(normals_vertice)
    norma_b = np.linalg.norm(normals_candidates, axis=1, keepdims=True)

    # Calcola il coseno dell'angolo
    coseno_angolo = prodotto_scalare / (norma_a * norma_b)

    # Calcola l'angolo in radianti
    angolo_radianti = np.arccos(np.clip(coseno_angolo, -1.0, 1.0))

    # Converti l'angolo in gradi
    angolo_gradi = - np.degrees(angolo_radianti) * 5

    # preleva i vertici della curva in un unico array
    maps_curve = [[filtered_edges[n_edge][i][0], filtered_edges[n_edge][i][1]]  for i in range(len(filtered_edges[n_edge]))]
    map1 = np.array(maps_curve)[:, 0].tolist()
    map2 = np.array(maps_curve)[:, 1].tolist()
    curve = list(set(map1 + map2))

    # rimuove i vertici della set_vertices_in_curves
    curve = list(set(list(set_vertices_in_curves.keys())).difference(curve))
    curve = np.array(curve)
    arr1 = np.array(vertices_real)[curve]
    arr2 = np.array(vertices_real)[candidates]
    diff = arr1[:, np.newaxis, :] - arr2
    # calcola la distanza tra il vertice e i vertici della curva
    distanze = np.linalg.norm(diff, axis=2)
    dist_min = np.min(distanze, axis=0).reshape(-1,1) * 200


    cost1 = np.sqrt(k1**2 + k2**2)
    cost1 = cost1.reshape(-1, 1) * 7

    cost2 = np.linalg.norm(np.sqrt(t1**2 + t2**2), axis=1, keepdims=True)
    
    #cost = cost1 + cost2 + angolo_gradi + dist_min
    #cost = cost1 + angolo_gradi + dist_min
    cost = cost1+dist_min+angolo_gradi
    #cost = dist_min

    # get the index of the minimum cost
    idx = np.argmin(cost)

    cost = cost[idx][0]
    candidate = candidates[idx]

    return candidate, cost

set_vertices_in_curves = {}
n_curve = 0
for curve in filtered_edges:
    for edge in curve:
        if edge[0] in set_vertices_in_curves:
            if n_curve not in set_vertices_in_curves[edge[0]]:
                set_vertices_in_curves[edge[0]].append(n_curve)
        else:
            set_vertices_in_curves[edge[0]] = [n_curve]
        if edge[1] in set_vertices_in_curves:
            if n_curve not in set_vertices_in_curves[edge[1]]:
                set_vertices_in_curves[edge[1]].append(n_curve)
        else:
            set_vertices_in_curves[edge[1]] = [n_curve]
    n_curve += 1

check_curves_first = [i for i in range(len(filtered_edges))]
check_curves_last = [i for i in range(len(filtered_edges))]

set_candidates = []

for n_edge in check_curves_last:
        
    curve = filtered_edges[n_edge]

    last_vertice = curve[-1][1]
    last_neighbors = one_ring_neighborhood(last_vertice, n_edge)

    if len(last_neighbors) == 0:
        continue
    
    new_candidate, cost = calculate_cost(last_vertice, last_neighbors, n_edge)
    set_candidates.append((cost, new_candidate, n_edge, 1))



for n_edge in check_curves_first:
    
    curve = filtered_edges[n_edge]
    first_vertice = curve[0][0]
    last_neighbors = one_ring_neighborhood(first_vertice, n_edge)

    if len(last_neighbors) == 0:
        continue

    new_candidate, cost = calculate_cost(first_vertice, last_neighbors, n_edge)
    set_candidates.append((cost, new_candidate, n_edge, 0))
        
it = 0
while len(check_curves_first) > 0 or len(check_curves_last) > 0:
    it += 1
    if len(set_candidates) == 0:
        break

    # sort the candidates in ascending order of cost
    set_candidates = sorted(set_candidates, key=lambda x: x[0])

    # preleva il primo elemento della lista dei candidati
    _, candidate, n_edge, direction = set_candidates[0]
    set_candidates.pop(0)

    # aggiunge il vertice alla curva
    if direction == 1:
        filtered_edges[n_edge].append([filtered_edges[n_edge][-1][1], candidate])
    else:
        filtered_edges[n_edge].insert(0, [candidate, filtered_edges[n_edge][0][0]])

    if candidate in set_vertices_in_curves:
        if n_edge not in set_vertices_in_curves[candidate]:
            
            #check if the candidate is in the first or last position of the curve
            for curves in set_vertices_in_curves[candidate]:
                
                if curves in check_curves_first:
                    if (filtered_edges[curves][0][0] == candidate):
                        check_curves_first.remove(curves)

                        # remove also from set_candidates
                        for i in range(len(set_candidates)):
                            if set_candidates[i][2] == curves and set_candidates[i][3] == 0:
                                set_candidates.pop(i)
                                break

                if curves in check_curves_last:
                    if (filtered_edges[curves][-1][1] == candidate):
                        check_curves_last.remove(curves)

                        # remove also from set_candidates
                        for i in range(len(set_candidates)):
                            if set_candidates[i][2] == curves and set_candidates[i][3] == 1:
                                set_candidates.pop(i)
                                break
                
            set_vertices_in_curves[candidate].append(n_edge)
            if direction == 1:
                check_curves_last.remove(n_edge)
            else:
                check_curves_first.remove(n_edge)
            continue
    else:
        set_vertices_in_curves[candidate] = [n_edge]

    # calculate the cost of the new edge
    neigbors = one_ring_neighborhood(candidate, n_edge)

    if(len(neigbors) > 0):
        new_candidate, cost = calculate_cost(candidate, neigbors, n_edge)
        set_candidates.append((cost, new_candidate, n_edge, direction))

# get the vertices with more than one curve
idx_vertices_to_split = [vertice for vertice in set_vertices_in_curves.keys() if len(set_vertices_in_curves[vertice]) > 1]
vertices_to_split = vertices_real[idx_vertices_to_split]


# merge the vertices if they are close
def merge_vertices(idx_vertices_to_split, vertices_to_split):
    global filtered_edges
    tol = 0.15

    # calculate the distance between each vertice and all the others
    diff = vertices_to_split[:, np.newaxis, :] - vertices_to_split
    distanze = np.linalg.norm(diff, axis=2)

    # get the index of the vertices to merge
    idx = np.argwhere((distanze < tol) & (distanze > 0))

    # remove the duplicates (i.e. [1, 2] and [2, 1])
    idx = [i for i in idx if i[0] < i[1]]

    # given the indexes of the vertices to remove, remove the rows from vertices_to_split

    # create a set of the indexes from 1 to vertices_to_split.shape[0]
    total_indexes = set([i for i in range(vertices_to_split.shape[0])])
    idx_to_remove = set(np.array(idx)[:, 1])
    indici_da_mantenere = list(total_indexes.difference(idx_to_remove))
    vertices_to_split = vertices_to_split[indici_da_mantenere]


    
    # merge the vertices
    for i in range(len(idx)):
        # in filtered_edges vertice 1 diventa vertice 2
        for idx_curve in range(len(filtered_edges)):
            for idx_edge in range(len(filtered_edges[idx_curve])):
                if filtered_edges[idx_curve][idx_edge][0] == idx_vertices_to_split[idx[i][1]]:
                    filtered_edges[idx_curve][idx_edge][0] = idx_vertices_to_split[idx[i][0]]
                if filtered_edges[idx_curve][idx_edge][1] == idx_vertices_to_split[idx[i][1]]:
                    filtered_edges[idx_curve][idx_edge][1] = idx_vertices_to_split[idx[i][0]]
    
    return vertices_to_split
            
vertices_to_split = merge_vertices(idx_vertices_to_split, vertices_to_split)

# rimuove tutti i filtered edges in cui vertice iniziale e finale sono uguali
filtered_edges = [curve for curve in filtered_edges if curve[0][0] != curve[-1][1]]


# unisce i filtered_edges in un unico array salvando dei colori diversi per ogni curva
edges2 = []
colors = []
for curve in filtered_edges:
    color = np.random.rand(3)

    for i in range(len(curve)):
        edges2.append([curve[i][0], curve[i][1]])
        colors.append(color)

# check se each elements in edges e edges2 sono uguali
edges_print = []
colors_print = []
for i in range(len(edges2)):
    edges_print.append(edges2[i])
    colors_print.append(colors[i])

#vertices_print = []
#for vertice in vertices:
#    vertices_print.append(vertices_real[mapping[vertices.index(vertice)]])

# Creiamo un LineSet usando Open3D
lineset = open3d.geometry.LineSet()

# Aggiungiamo i punti al LineSet
lineset.points = open3d.utility.Vector3dVector(vertices_real)

# Aggiungiamo gli edges al LineSet
lineset.lines = open3d.utility.Vector2iVector(edges_print)


# Creare un oggetto PointCloud di Open3D
#point_cloud = open3d.geometry.PointCloud()
#point_cloud.points = open3d.utility.Vector3dVector(np.array(vertices_to_split))

# Aggiungiamo i colori al LineSet
lineset.colors = open3d.utility.Vector3dVector(colors_print)

# Visualizziamo il LineSet
open3d.visualization.draw_geometries([lineset])