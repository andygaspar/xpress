import copy

import xpress as xp
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

black = "#000000"
blue = "#1f78b4"
green = "#008000"
red = "#FF0000"

square = "s"
dot = "o"
cross = "x"
shapes = [dot, cross, square]

xp.controls.outputlog = 0


class Node:
    def __init__(self, pb: xp.problem,  parent=None, l_child=None, r_child=None,
                 var_idx=None, var_value=None, is_left=False):
        self.pb = pb.copy()
        self.objVal = - math.inf
        self.tree_idx = None
        self.parent = parent
        self.lChild = l_child
        self.rChild = r_child

        self.varIdx = var_idx
        self.varValue = var_value
        self.isLeft = is_left

        self.is_solved = False

        self.is_integer = False
        self.is_infeasible = False
        self.is_pruned = False
        self.is_optimal = False

    def solve(self):
        self.pb.solve()
        self.is_solved = True
        if self.pb.getProbStatus() == 2:  # infeasible
            self.is_infeasible = True
            return False, True

        self.objVal = self.pb.getObjVal()

        if self.has_integer_solution():
            self.is_integer = True
            return True, False

        return False, False

    @staticmethod
    def get_first_non_integer_var(x):
        for i in range(len(x)):
            if not x[i].is_integer():
                return i

    def make_sub_problems(self):
        x = self.pb.getSolution()
        index = self.get_first_non_integer_var(x)

        left_pb = self.pb.copy()
        left_vars = left_pb.getVariable()
        left_pb.addConstraint(left_vars[index] <= int(x[index]))
        left_node = Node(left_pb, parent=self, var_idx=index, var_value=int(x[index]), is_left=True)
        self.lChild = left_node

        right_pb = self.pb.copy()
        right_vars = right_pb.getVariable()
        right_pb.addConstraint(right_vars[index] >= int(x[index]) + 1)
        right_node = Node(right_pb, parent=self, var_idx=index, var_value=int(x[index] + 1))
        self.rChild = right_node

        return left_node, right_node

    def has_integer_solution(self):
        x = self.pb.getSolution()
        for x_i in x:
            if not x_i.is_integer():
                return False
        return True


def is_optimal(lb_node, active_nodes_list):
    for node in active_nodes_list:
        if lb_node.objVal < node.objVal:
            return False
    return True


def fill_tree(T, node: Node, labels):
    shape = cross if node.is_infeasible else square if node.is_pruned else dot
    color = red if node.is_infeasible or node.is_pruned else green if node.is_optimal \
        else blue if node.is_integer else black

    node.tree_idx = T.number_of_nodes()
    T.add_node(node.tree_idx, shape=shape, color=color)
    T.add_edge(node.parent.tree_idx, node.tree_idx)
    inequality = "<=" if node.isLeft else ">="
    label = "x_" + str(node.varIdx) + inequality + str(node.varValue)

    if node.is_solved:
        solution = "x=" + str(np.round(node.pb.getSolution(), 2)) if not node.is_infeasible else "INFEASIBLE"
        obj = "\nobj=" + str(np.round(node.objVal, 2)) if not node.is_infeasible else ""
        label += "\n" + solution + obj

    labels[node.tree_idx] = label

    if node.lChild is not None:
        fill_tree(T, node.lChild, labels)
    if node.rChild is not None:
        fill_tree(T, node.rChild, labels)


def show_tree(root_node):
    T = nx.Graph()
    root_node.tree_idx = 0
    solution = "x=" + str(np.round(root_node.pb.getSolution(), 2)) if not root_node.is_infeasible else "INFEASIBLE"
    obj = "\nobj=" + str(np.round(root_node.objVal, 2)) if not root_node.is_infeasible else ""
    labels = {0: 'Root\n' + solution + obj}

    T.add_node(root_node.tree_idx, shape=dot if not root_node.is_infeasible else cross,
               color=green if root_node.is_integer else red if root_node.is_infeasible else black)

    if root_node.lChild is not None:
        fill_tree(T, root_node.lChild, labels)
    if root_node.rChild is not None:
        fill_tree(T, root_node.rChild, labels)
    pos = graphviz_layout(T, prog="dot")


    plt.figure(3, figsize=(12, 12))
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.80
    plt.xlim(x_min - x_margin, x_max + x_margin)

    label_pos = copy.deepcopy(pos)
    print(x_margin)
    for key in label_pos.keys():
        offset = 0.3 if x_margin < 20 else 30
        label_pos[key] = ((label_pos[key][0] + offset), label_pos[key][1])

    node_shapes = dict(zip(shapes, [[node for node in T.nodes(data=True) if node[1]["shape"] == shape]
                                    for shape in shapes]))

    for shape in shapes:
        nodes = [node[0] for node in node_shapes[shape]]
        colors = [node[1]["color"] for node in node_shapes[shape]]
        nx.draw_networkx_nodes(T, pos, node_color=colors, node_shape=shape, nodelist=nodes)

    nx.draw_networkx_edges(T, pos)
    nx.draw_networkx_labels(T, label_pos, labels, horizontalalignment="center", font_size=7)

    cols_dict = {'Non int': black, 'Int':  blue, 'Pruned': red, 'Inf': red, 'Optimal': green}

    # The following two lines generate custom fake lines that will be used as legend entries:
    markers = [plt.Line2D([0, 0], [0, 0], color=cols_dict[key],
                          marker=cross if key == 'Inf' else square if key == 'Pruned' else dot, linestyle='')
               for key in cols_dict.keys()]
    plt.legend(markers, cols_dict.keys(), numpoints=1)

    plt.show()


def bb(A, b):

    pb = xp.problem()
    x_lp = np.array([xp.var(vartype=xp.continuous) for _ in range(A.shape[1])])
    pb.addVariable(x_lp)
    pb.addConstraint(xp.Dot(A, x_lp) <= b)
    pb.setObjective(xp.Dot(c, x_lp), sense=xp.maximize)

    lb_node = Node(xp.problem(), -1)

    root_node = Node(pb)

    is_integer, is_infeasible = root_node.solve()

    show_tree(root_node)

    if is_integer:
        return root_node.pb.getObjVal(), root_node.pb.getSolution()
    elif is_infeasible:
        return False, False

    l_node, r_node = root_node.make_sub_problems()

    active_nodes = [l_node, r_node]

    while True:
        show_tree(root_node)
        not_pruned_nodes = []
        for node in active_nodes:
            is_integer, is_infeasible = node.solve()
            if is_integer:
                if node.objVal > lb_node.objVal:
                    lb_node = node
            elif not is_infeasible and node.objVal > lb_node.objVal:
                not_pruned_nodes.append(node)
            else:
                node.is_pruned = True
            # # inp = input()


        if is_optimal(lb_node, not_pruned_nodes):
            lb_node.is_optimal = True
            show_tree(root_node)
            return lb_node.pb.getObjVal(), lb_node.pb.getSolution()

        active_nodes = []
        for node in not_pruned_nodes:
            l_node, r_node = node.make_sub_problems()
            active_nodes.append(l_node)
            active_nodes.append(r_node)


c = np.array([1., 0.64])
A = np.array([[50, 31],
              [-3, 2]])
b = np.array([250, 4])

o, sol = bb(A, b)

print(o, sol)
