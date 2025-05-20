# TTP
## Descripción 
Este repositorio alberga un script que implementa las dos formulaciones del TTP basta con contar con Python 3.7 o superior, una licencia activa de Gurobi y las librerías necesarias (gurobipy, numpy, networkx y matplotlib).   

El programa resolverá automáticamente ambos modelos (“SEC” y “compacto”) con la semilla, el número de clientes y los costes definidos en el propio archivo, mostrará por consola la asignación de nodos, el valor de la función objetivo y los tiempos de cómputo, y generará dos gráficos en los que se ilustran el tour (en rojo) y el árbol (en verde). Si deseas probar otros escenarios, modifica al inicio de TTP_TFM-CarolinaSanchez.py los parámetros nclients, np.random.seed(...) o el diccionario d.
