import xpress as xp
import numpy as np
import copy
import math

xp.controls.outputlog = 0


class Node:
    def __init__(self, num_vars, solution=None, idxs=None):
        self.solution = solution if solution is not None else np.zeros(num_vars)
        self.numVars = num_vars
        self.idxs = idxs
        self.lastIdx = None
        self.ub = None
        self.lb = None

        self.infeasible = False

    def fill(self, costs, weights, W):
        idxs_to_check = list(set(range(self.numVars))-set(self.idxs))
        last_idx = None
        for i in idxs_to_check:
            self.solution[i] = 1
            last_idx = i
            if np.dot(weights, self.solution) >= W:
                break

        if last_idx == self.idxs[0]:
            self.infeasible = True
            return

        self.lb = np.dot(costs, self.solution)
        if last_idx != idxs_to_check[-1]:
            self.solution[last_idx] = 0
            self.ub = self.lb + (W - np.dot(weights, self.solution)) * costs[last_idx]/weights[last_idx]
            self.lastIdx = last_idx
        else:
            self.ub = self.lb

    def make_sub_problems(self):
        left_sol = self.solution.copy()
        left_sol[self.lastIdx] = 0
        idxs = copy.copy(self.lastIdx)
        idxs.append(self.lastIdx)
        left_node = Node(self.numVars, left_sol, idxs)

        right = self.solution.copy()
        right[self.lastIdx] = 1
        right_node = Node(self.numVars, right, idxs)

        return left_node, right_node


def is_optimal(lb_node, active_nodes_list):
    for node in active_nodes_list:
        if lb_node.objVal < node.objVal:
            return False
    return True


def bb(A, b):

    lb_node = Node(xp.problem())

    root_node = Node(pb)
    is_integer, is_infeasible = root_node.solve()
    if is_integer:
        return root_node.pb.getObjVal(), root_node.pb.getSolution()
    elif is_infeasible:
        return False, False

    l_node, r_node = root_node.make_sub_problems()

    active_nodes = [l_node, r_node]

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




