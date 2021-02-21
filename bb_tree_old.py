import xpress as xp
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

xp.controls.outputlog = 0




class Node:
    def __init__(self, pb: xp.problem, tree_idx):
        self.pb = pb.copy()
        self.objVal = - math.inf
        self.tree_idx = tree_idx

    def solve(self):
        self.pb.solve()
        if self.pb.getProbStatus() == 2:  # infeasible
            return False, True

        self.objVal = self.pb.getObjVal()

        if self.has_integer_solution():
            return True, False

        return False, False

    @staticmethod
    def get_first_non_integer_var(x):
        for i in range(len(x)):
            if not x[i].is_integer():
                return i

    def make_sub_problems(self, tree_idx):
        x = self.pb.getSolution()
        index = self.get_first_non_integer_var(x)

        left_pb = self.pb.copy()
        left_vars = left_pb.getVariable()
        left_pb.addConstraint(left_vars[index] <= int(x[index]))
        left_node = Node(left_pb, tree_idx+1)

        right_pb = self.pb.copy()
        right_vars = right_pb.getVariable()
        right_pb.addConstraint(right_vars[index] >= int(x[index]) + 1)
        right_node = Node(right_pb, tree_idx + 2)

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




def bb(A, b):
    T = nx.Graph()
    tree_idx = 0

    pb = xp.problem()
    x_lp = np.array([xp.var(vartype=xp.continuous) for _ in range(A.shape[1])])
    pb.addVariable(x_lp)
    pb.addConstraint(xp.Dot(A, x_lp) <= b )
    pb.setObjective(xp.Dot(c, x_lp), sense=xp.maximize)

    lb_node = Node(xp.problem(), -1)

    root_node = Node(pb, tree_idx)
    T.add_node(tree_idx)

    is_integer, is_infeasible = root_node.solve()
    if is_integer:
        return root_node.pb.getObjVal(), root_node.pb.getSolution()
    elif is_infeasible:
        return False, False

    l_node, r_node = root_node.make_sub_problems(tree_idx)

    active_nodes = [l_node, r_node]
    T.add_node(l_node.tree_idx)
    T.add_node(r_node.tree_idx)
    T.add_edge(root_node.tree_idx, l_node.tree_idx)
    T.add_edge(root_node.tree_idx, r_node.tree_idx)
    tree_idx += 3

    while True:

        not_pruned_nodes = []
        for node in active_nodes:
            is_integer, is_infeasible = node.solve()
            if is_integer:
                if node.objVal > lb_node.objVal:
                    lb_node = node
            elif not is_infeasible:
                not_pruned_nodes.append(node)

        if is_optimal(lb_node, not_pruned_nodes):
            pos = graphviz_layout(T, prog="dot")
            print(pos)
            labels = pos
            print(labels)
            nx.draw(T, pos)
            nx.draw_networkx_labels(T, pos, labels)
            plt.show()
            return lb_node.pb.getObjVal(), lb_node.pb.getSolution()

        active_nodes = []
        for node in not_pruned_nodes:
            l_node, r_node = node.make_sub_problems(tree_idx)
            active_nodes.append(l_node)
            active_nodes.append(r_node)
            T.add_node(l_node.tree_idx)
            T.add_node(r_node.tree_idx)
            T.add_edge(node.tree_idx, l_node.tree_idx)
            T.add_edge(node.tree_idx, r_node.tree_idx)
            tree_idx += 3


c = np.array([1., 0.64])
A = np.array([[50, 31],
             [-3, 2]])
b = np.array([250, 4])

o, sol = bb(A, b)

print(o, sol)




