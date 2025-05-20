import gurobipy as gp
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

env = gp.Env(empty = True)
env.setParam('LogToConsole', 0)
env.start()

## Datos y parámetros del TTP
EPS = 1e-6 
np.random.seed(622)
nclients = 15

# Definimos nodos: 1 = depósito tour, 2 = depósito árbol, resto = clientes
nodes = list(range(1, nclients + 3))
clients = list(range(3, nclients + 3))
tour_dep = 1
tree_dep = 2
coords = {i: (np.random.randint(0, 10, 2)) for i in nodes}

# Conjunto de aristas (E) y arcos (A)
E = [(i, j) for i in nodes for j in nodes if i < j]   # Conjunto de aristas
A = [(i, j) for i in nodes for j in nodes if i != j]   # Conjunto de arcos

# Costo extra de activación de nodo (d) y distancia euclídea (c)
d = {i: 0.0 for i in clients} 
c = {(i, j): math.dist(coords[i], coords[j]) for (i, j) in E} 

## Funciones auxiliares
# Mostramos resultados de asignación y valor objetivo en pantalla
def print_solution_info(model, sol):
    y, x, obj_value, time = sol
    print("Asignación de nodos con el modelo ", model, " : ")
    print("  TOUR  =", [i for i in clients if y[i] == 1])
    print("  ÁRBOL =", [i for i in clients if y[i] == 0])
    print("Valor de la función objetivo:", obj_value)
    print("Tiempo:", time, "segundos")
    print("Número de cortes:", counterSECint, " ", counterSECfrac, " ",  counterTOURint, " ",  counterTOURfrac, " ",  counterTREEint, " ",  counterTREEfrac)

# Dibujamos el tour (rojo) y árbol (verde)
def draw_solution(model, sol):
    y, x, obj_value, time = sol
    title = model + " Costo: " + str(obj_value) + " Tiempo: " + str(time) + " segundos."
    G = nx.Graph()
    G.add_nodes_from(nodes)
    pos = {i: coords[i] for i in nodes}

    # Separamos aristas según tipo de conexión
    tour_nodes = set([tour_dep] + [i for i in clients if y[i] == 1])
    tree_nodes = set([tree_dep] + [i for i in clients if y[i] == 0])
    t_edges = [e for e in x if e[0] in tour_nodes and e[1] in tour_nodes]
    a_edges = [e for e in x if e[0] in tree_nodes and e[1] in tree_nodes]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size = 500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist = t_edges, edge_color = 'red', width = 2, label ='Tour')
    nx.draw_networkx_edges(G, pos, edgelist = a_edges, edge_color = 'green', width = 2, label ='Árbol')
    plt.title(title)
    plt.axis('off')
    plt.legend()
    plt.show()

# Función principal que resuelve ambas formulaciones
def solve(model):
    global counterSECint, counterSECfrac, counterTOURint, counterTOURfrac, counterTREEint, counterTREEfrac
    counterSECint = counterSECfrac = counterTOURint = counterTOURfrac = counterTREEint = counterTREEfrac = 0

    # Creamos modelo y variables de decisión
    m = gp.Model(model, env = env)
    m.Params.OutputFlag = 0
    m.Params.LogToConsole = 0 

    x = m.addVars(E, vtype=gp.GRB.BINARY, name="x")
    y = m.addVars(nodes, vtype=gp.GRB.BINARY, name="y")

    # Función objetivo
    m.setObjective(gp.quicksum(c[(i, j)] * x[(i, j)] for (i, j) in E) + gp.quicksum(d[i] * y[i] for i in clients), gp.GRB.MINIMIZE)

    # Forzamos depósitos
    m.addConstr(y[tour_dep] == 1) # Nodo 1 en tour
    m.addConstr(y[tree_dep] == 0) # Nodo 2 en tour

    # Evitamos las aristas que unan nodos que corresponden a recorridos diferentes (tour y árbol)
    for (i, j) in E:
        m.addConstr(x[(i, j)] <= y[i] + (1 - y[j]))
        m.addConstr(x[(i, j)] <= y[j] + (1 - y[i]))
        
    # Cotas de grado máximas y mínimas
    for k in clients:
        m.addConstr(gp.quicksum(x[(u,v)] for (u,v) in E if u == k or v == k) >= 1 + y[k])
        m.addConstr(gp.quicksum(x[(u,v)] for (u,v) in E if u == k or v == k) <= nclients - (nclients - 2) * y[k])

    # Forzamos grado exacto 2 en depósito de tour
    m.addConstr(gp.quicksum(x[(u, v)] for (u, v) in E if u == tour_dep or v == tour_dep) == 2 * y[tour_dep])

    # Formulación compacta
    if model == "compacto":

        # Variables de flujo
        f = {(k, a): m.addVar(vtype=gp.GRB.CONTINUOUS, lb = 0) for k in clients for a in A}
        g = {(k, a): m.addVar(vtype=gp.GRB.CONTINUOUS, lb = 0) for k in clients for a in A}

        for k in clients:
            for i in nodes:
                if i == k:
                    rhsf, rhsg = y[k], 1 - y[k]
                elif i == tour_dep:
                    rhsf, rhsg = -y[k], 0
                elif i == tree_dep:
                    rhsf, rhsg = 0, y[k] - 1
                else:
                    rhsf = rhsg = 0

                # Balance de flujo
                m.addConstr(gp.quicksum(f[(k, (j, i))] - f[(k, (i, j))] for j in nodes if j != i) == rhsf)
                m.addConstr(gp.quicksum(g[(k, (j, i))] - g[(k, (i, j))] for j in nodes if j != i) == rhsg)
            
            for (i, j) in E:
                m.addConstr(f[(k, (i, j))] + f[(k, (j, i))] + g[(k, (i, j))] + g[(k, (j, i))] <= x[(i, j)])
        
        m.optimize()

    # Formulación con SEC
    elif model == "SEC":
        def cutting(model, where):
            global counterSECint, counterSECfrac, counterTOURint, counterTOURfrac, counterTREEint, counterTREEfrac

            # Extraemos soluciones enteras
            if where == gp.GRB.Callback.MIPSOL:
                xvals = model.cbGetSolution(x)
                yvals = model.cbGetSolution(y)

                # Construimos un grafo no dirigido
                G = nx.Graph() 
                G.add_nodes_from(nodes) 
                G.add_edges_from((i, j) for (i, j) in E if xvals[(i, j)] > EPS) 
                # Hallamos componentes conexas
                Components = list(nx.connected_components(G))

                if len(Components) > 2: 
                    for S in Components:
                        if tour_dep not in S and tree_dep not in S:
                            # Corte SEC entero
                            if sum(xvals[u, j] for u in S for j in S if u < j) > len(S) - 1 + EPS:
                                model.cbLazy(gp.quicksum(x[(u, v)] for u in S for v in S if u < v) <= len(S) - 1)
                                counterSECint += 1

                            # Corte de conectividad del tour
                            T = set(clients) - S
                            T.add(tour_dep)
                            for i in S:
                                if sum(xvals[(u, v)] for u in S for v in T if u < v) + sum(xvals[(v, u)] for u in S for v in T if u > v) < 2 * yvals[i] - EPS:
                                    model.cbLazy(gp.quicksum(x[(u, v)] for u in S for v in T if u < v) + gp.quicksum(x[(v, u)] for u in S for v in T if u > v) >= 2 * y[i])
                                    counterTOURint += 1
                            
                            # Corte de conectividad del árbol
                            T = set(clients) - S
                            T.add(tree_dep)
                            for i in S:
                                if sum(xvals[(u, v)] for u in S for v in T if u < v) + sum(xvals[(v, u)] for u in S for v in T if u > v) < 1 - yvals[i] - EPS:
                                    model.cbLazy(gp.quicksum(x[(u, v)] for u in S for v in T if u < v) + gp.quicksum(x[(v, u)] for u in S for v in T if u > v) >= 1 - y[i])
                                    counterTREEint += 1

            # Relajación fraccional
            elif where == gp.GRB.Callback.MIPNODE:
                try:
                    xvals = model.cbGetNodeRel(x)
                    yvals = model.cbGetNodeRel(y)
                except gp.GurobiError:
                    return  # Sale de la función si no se puede acceder a la relajación de nodos

                G = nx.Graph()
                G.add_nodes_from(nodes) # para parte SEC fraccional
                d = {i: 0.0 for i in clients} # Acumula el flujo total incidente en i
                for edge, val in xvals.items():
                    if val > EPS:
                        u, v = edge
                        #en G repartimos la capacidad a la mitad para cada dirección
                        if u != tour_dep and u != tree_dep and v != tour_dep and v != tree_dep:
                            G.add_edge(u, v, capacity = val/2)
                            G.add_edge(v, u, capacity = val/2)
                            d[u] += val
                            d[v] += val

                for i in clients:
                    # Corte SEC fraccional 
                    G.add_edge(tour_dep, i, capacity = 1) # arcos (1 -> i) con capacidad = 1. 
                    G.add_edge(i, tree_dep, capacity = d[i]/2)

                for i in clients: 
                    G[tour_dep][i]['capacity'] += nclients
                    cut_value, (S, T) = nx.minimum_cut(G, tour_dep, tree_dep, capacity ='capacity')
                    S.remove(tour_dep)
                    if cut_value < nclients - EPS:
                        model.cbCut(gp.quicksum(x[(u, v)] for u in S for v in S if u < v) <= len(S) - 1)
                        counterSECfrac += 1        
                    G[tour_dep][i]['capacity'] -= nclients

                G_con = nx.Graph() # para cortes de tour y árbol
                G_con.add_nodes_from(nodes)
                for edge, val in xvals.items():
                    if val > EPS:
                        u, v = edge
                        # en G_con ponemos capacidad completa
                        G_con.add_edge(u, v, capacity = val)
                        G_con.add_edge(v, u, capacity = val)

                for i in clients:
                    # Corte de tour fraccional
                    if 2 * yvals[i] > EPS:
                        cut_value, (S, T) = nx.minimum_cut(G_con, tour_dep, i, capacity ='capacity')
                        if cut_value < 2 * yvals[i] - EPS:
                            expression = gp.LinExpr() 
                            for u in S:
                                for v in T:
                                    if u < v:
                                        expression += x[(u, v)]
                                    else:
                                        expression += x[(v, u)]
                            model.cbCut(expression >= 2 * y[i])
                            counterTOURfrac += 1

                    # Corte de árbol fraccional
                    if 1 - yvals[i] > EPS:
                        cut_value, (S, T) = nx.minimum_cut(G_con, tree_dep, i, capacity ='capacity')
                        if cut_value < 1 - yvals[i] - EPS:
                            expression = gp.LinExpr()  
                            for u in S:
                                for v in T:
                                    if u < v:
                                        expression += x[(u, v)]
                                    else:
                                        expression += x[(v, u)]
                            model.cbCut(expression >= 1 - y[i])
                            counterTREEfrac += 1

        m.Params.lazyConstraints = 1
        m.optimize(cutting)

    else:
        print("Modelo no soportado")

    if m.status == gp.GRB.OPTIMAL:
        return {i: int(y[i].X) for i in clients}, [e for e in E if x[e].X > 0.5], m.objVal, m.Runtime
    else:
        print("No se encontró solución óptima ; status = ", m.status)
        return None

# Resolvemos el problema TTP:
for model in ["SEC", "compacto"]:
    sol = solve(model)
    if sol is not None: 
        print_solution_info(model, sol)
        draw_solution(model, sol)
    else:
        print("No se encontró solución óptima para el modelo ", model)      