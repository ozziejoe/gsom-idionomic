"""
Growing Self-Organizing Map (GSOM)
==================================

A GSOM extends the Kohonen Self-Organizing Map with the ability to *grow* new
nodes where the map fits the data poorly. This makes the final map size
data-driven rather than fixed in advance, which is what we want for idionomic
analysis: the number of distinct person-types is discovered, not imposed.

This implementation uses a modified neighbourhood update that skips nodes in the
same row/column as the winner. It is the implementation used in the idionomic
pipeline (Ciarrochi et al.).

The class is intentionally dependency-light (numpy + scipy + bigtree) so it runs
unchanged in a notebook, on a server, or fully in-browser via Pyodide/stlite.
"""

import math

import numpy as np
import pandas as pd
import scipy.spatial
from bigtree import Node, find, tree_to_dataframe


class GSOM:
    """Growing Self-Organizing Map.

    Parameters
    ----------
    spread_factor : float
        Controls how readily the map grows. Lower values -> more growth (a
        larger map); higher values -> fewer nodes. Typical range 0.3-0.9.
    dimensions : int
        Number of input features (columns of the data matrix).
    learning_rate : float, default 0.3
        Initial learning rate for the growing phase.
    smooth_learning_factor : float, default 0.8
        Multiplier applied to the learning rate for the smoothing phase.
    max_radius : int, default 6
        Maximum neighbourhood radius.
    FD : float, default 0.1
        Factor of distribution -- how error is spread to neighbours on growth.
    """

    def __init__(self, spread_factor, dimensions, distance="euclidean",
                 initialize="random", learning_rate=0.3,
                 smooth_learning_factor=0.8, max_radius=6, FD=0.1, r=3.8,
                 alpha=0.9, initial_node_size=30000):
        self.initial_node_size = initial_node_size
        self.node_count = 0
        self.map = {}
        self.node_list = np.zeros((self.initial_node_size, dimensions))
        self.node_coordinate = np.zeros((self.initial_node_size, 2))
        self.node_errors = np.zeros(self.initial_node_size, dtype=np.longdouble)
        self.spred_factor = spread_factor
        self.groth_threshold = -dimensions * math.log(self.spred_factor)
        self.FD = FD
        self.R = r
        self.ALPHA = alpha
        self.dimentions = dimensions
        self.distance = distance
        self.initialize = initialize
        self.learning_rate = learning_rate
        self.smooth_learning_factor = smooth_learning_factor
        self.max_radius = max_radius
        self.node_labels = None
        self.output = None
        self.predictive = None
        self.active = None
        self.sequence_weights = None
        self.path_tree = {}
        self.initialize_GSOM()

    # ------------------------------------------------------------------ setup
    def initialize_GSOM(self):
        self.path_tree = Node("root", x=0.01, y=0.01, node_number=-1, distance=0)
        for x, y in [(1, 1), (1, 0), (0, 1), (0, 0)]:
            self.insert_node_with_weights(x, y)

    def insert_new_node(self, x, y, weights, parent_node=None):
        if self.node_count > self.initial_node_size:
            raise MemoryError("GSOM node size out of bound")
        self.map[(x, y)] = self.node_count
        self.node_list[self.node_count] = weights
        self.node_coordinate[self.node_count][0] = x
        self.node_coordinate[self.node_count][1] = y

        distance_from_parent = 0
        new_node = Node(str(self.node_count), x=x, y=y,
                        node_number=self.node_count, distance=distance_from_parent)

        if parent_node is not None:
            if (parent_node.x, parent_node.y) in self.map:
                distance_from_parent = scipy.spatial.distance.cdist(
                    weights.reshape(1, -1),
                    self.node_list[self.map[(parent_node.x, parent_node.y)]].reshape(1, -1),
                    self.distance,
                )
                new_node.distance = distance_from_parent[0][0]
            new_node.parent = parent_node
        else:
            raise ValueError("Parent node is not provided")
        self.node_count += 1

    def insert_node_with_weights(self, x, y):
        if self.initialize == "random":
            node_weights = np.random.rand(self.dimentions)
        else:
            raise NotImplementedError("Initialization method not supported")
        self.insert_new_node(x, y, node_weights, parent_node=self.path_tree)

    # ------------------------------------------------------------- parameters
    def _get_learning_rate(self, prev_learning_rate):
        return self.ALPHA * (1 - (self.R / self.node_count)) * prev_learning_rate

    def _get_neighbourhood_radius(self, total_iteration, iteration):
        time_constant = total_iteration / math.log(self.max_radius)
        return self.max_radius * math.exp(-iteration / time_constant)

    # ----------------------------------------------------------- node growing
    def _new_weights_for_new_node_in_middle(self, winnerx, winnery, next_nodex, next_nodey):
        return (self.node_list[self.map[(winnerx, winnery)]]
                + self.node_list[self.map[(next_nodex, next_nodey)]]) * 0.5

    def _new_weights_for_new_node_on_one_side(self, winnerx, winnery, next_nodex, next_nodey):
        return (2 * self.node_list[self.map[(winnerx, winnery)]]
                - self.node_list[self.map[(next_nodex, next_nodey)]])

    def _new_weights_for_new_node_one_older_neighbour(self, winnerx, winnery):
        return np.full(self.dimentions,
                       (max(self.node_list[self.map[(winnerx, winnery)]])
                        + min(self.node_list[self.map[(winnerx, winnery)]])) / 2)

    def grow_node(self, wx, wy, x, y, side):
        if (x, y) in self.map:
            return
        if side == 0:  # left
            if (x - 1, y) in self.map:
                weights = self._new_weights_for_new_node_in_middle(wx, wy, x - 1, y)
            elif (wx + 1, wy) in self.map:
                weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx + 1, wy)
            elif (wx, wy + 1) in self.map:
                weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy + 1)
            elif (wx, wy - 1) in self.map:
                weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy - 1)
            else:
                weights = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
        elif side == 1:  # right
            if (x + 1, y) in self.map:
                weights = self._new_weights_for_new_node_in_middle(wx, wy, x + 1, y)
            elif (wx - 1, wy) in self.map:
                weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx - 1, wy)
            elif (wx, wy + 1) in self.map:
                weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy + 1)
            elif (wx, wy - 1) in self.map:
                weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy - 1)
            else:
                weights = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
        elif side == 2:  # top
            if (x, y + 1) in self.map:
                weights = self._new_weights_for_new_node_in_middle(wx, wy, x, y + 1)
            elif (wx, wy - 1) in self.map:
                weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy - 1)
            elif (wx + 1, wy) in self.map:
                weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx + 1, wy)
            elif (wx - 1, wy) in self.map:
                weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx - 1, wy)
            else:
                weights = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
        elif side == 3:  # bottom
            if (x, y - 1) in self.map:
                weights = self._new_weights_for_new_node_in_middle(wx, wy, x, y - 1)
            elif (wx, wy + 1) in self.map:
                weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy + 1)
            elif (wx + 1, wy) in self.map:
                weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx + 1, wy)
            elif (wx - 1, wy) in self.map:
                weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx - 1, wy)
            else:
                weights = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
        else:
            raise ValueError("Invalid side specified")

        weights[weights < 0] = 0.0
        weights[weights > 1] = 1.0

        parent_node = find(self.path_tree, lambda node: node.x == wx and node.y == wy)
        self.insert_new_node(x, y, weights, parent_node=parent_node)

    def spread_wights(self, x, y):
        leftx, lefty = x - 1, y
        rightx, righty = x + 1, y
        topx, topy = x, y + 1
        bottomx, bottomy = x, y - 1
        self.node_errors[self.map[(x, y)]] = self.groth_threshold / 2
        self.node_errors[self.map[(leftx, lefty)]] *= (1 + self.FD)
        self.node_errors[self.map[(rightx, righty)]] *= (1 + self.FD)
        self.node_errors[self.map[(topx, topy)]] *= (1 + self.FD)
        self.node_errors[self.map[(bottomx, bottomy)]] *= (1 + self.FD)

    def adjust_wights(self, x, y, rmu_index):
        leftx, lefty = x - 1, y
        rightx, righty = x + 1, y
        topx, topy = x, y + 1
        bottomx, bottomy = x, y - 1
        if ((leftx, lefty) in self.map and (rightx, righty) in self.map
                and (topx, topy) in self.map and (bottomx, bottomy) in self.map):
            self.spread_wights(x, y)
        else:
            self.grow_node(x, y, leftx, lefty, 0)
            self.grow_node(x, y, rightx, righty, 1)
            self.grow_node(x, y, topx, topy, 2)
            self.grow_node(x, y, bottomx, bottomy, 3)
        self.node_errors[rmu_index] = self.groth_threshold / 2

    # ------------------------------------------------------------- training
    def winner_identification_and_neighbourhood_update(self, data_index, data, radius, learning_rate):
        out = scipy.spatial.distance.cdist(
            self.node_list[:self.node_count],
            data[data_index, :].reshape(1, self.dimentions),
            self.distance,
        )
        rmu_index = out.argmin()
        error_val = out.min()
        rmu_x = int(self.node_coordinate[rmu_index][0])
        rmu_y = int(self.node_coordinate[rmu_index][1])

        error = data[data_index] - self.node_list[rmu_index]
        self.node_list[self.map[(rmu_x, rmu_y)]] += learning_rate * error

        mask_size = round(radius)
        for i in range(rmu_x - mask_size, rmu_x + mask_size):
            for j in range(rmu_y - mask_size, rmu_y + mask_size):
                if (i, j) in self.map and (i != rmu_x and j != rmu_y):
                    error = self.node_list[rmu_index] - self.node_list[self.map[(i, j)]]
                    dist = (rmu_x - i) * (rmu_x - i) + (rmu_y - j) * (rmu_y - j)
                    eDistance = np.exp(-1.0 * dist / (2.0 * (radius * radius)))
                    self.node_list[self.map[(i, j)]] += learning_rate * eDistance * error
        return rmu_index, rmu_x, rmu_y, error_val

    def smooth(self, data, radius, learning_rate):
        for data_index in range(data.shape[0]):
            self.winner_identification_and_neighbourhood_update(data_index, data, radius, learning_rate)

    def grow(self, data, radius, learning_rate):
        for data_index in range(data.shape[0]):
            rmu_index, rmu_x, rmu_y, error_val = self.winner_identification_and_neighbourhood_update(
                data_index, data, radius, learning_rate
            )
            self.node_errors[rmu_index] += error_val
            if self.node_errors[rmu_index] > self.groth_threshold:
                self.adjust_wights(rmu_x, rmu_y, rmu_index)

    def fit(self, data, training_iterations, smooth_iterations, progress=None):
        """Train the map.

        Parameters
        ----------
        data : ndarray (n_samples, n_features)
        training_iterations, smooth_iterations : int
        progress : callable(fraction_done: float, label: str) or None
            Optional callback for UI progress bars (Streamlit, etc.).
        """
        total = max(1, training_iterations + smooth_iterations)
        done = 0

        current_learning_rate = self.learning_rate
        for i in range(training_iterations):
            radius_exp = self._get_neighbourhood_radius(training_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)
            self.grow(data, radius_exp, current_learning_rate)
            done += 1
            if progress:
                progress(done / total, "Growing")

        current_learning_rate = self.learning_rate * self.smooth_learning_factor
        for i in range(smooth_iterations):
            radius_exp = self._get_neighbourhood_radius(training_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)
            self.smooth(data, radius_exp, current_learning_rate)
            done += 1
            if progress:
                progress(done / total, "Smoothing")

        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data, self.distance)
        return out.argmin(axis=0)

    # -------------------------------------------------------------- predict
    def predict(self, data, index_col, label_col=None):
        weight_columns = list(data.columns.values)
        output_columns = [index_col]
        if label_col:
            weight_columns.remove(label_col)
            output_columns.append(label_col)
        weight_columns.remove(index_col)
        data_n = data[weight_columns].to_numpy()
        data_out = pd.DataFrame(data[output_columns])
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data_n, self.distance)
        data_out["output"] = out.argmin(axis=0)
        grp_output = data_out.groupby("output")
        dn = grp_output[index_col].apply(list).reset_index()
        dn = dn.set_index("output")
        if label_col:
            dn[label_col] = grp_output[label_col].apply(list)
        dn = dn.reset_index()
        dn["hit_count"] = dn[index_col].apply(lambda x: len(x))
        dn["x"] = dn["output"].apply(lambda x: self.node_coordinate[x, 0])
        dn["y"] = dn["output"].apply(lambda x: self.node_coordinate[x, 1])
        self.node_labels = dn
        self.output = data_out
        return self.node_labels

    def get_paths(self):
        paths = []
        paths.extend(self.path_tree.get_paths())
        return paths

    def skeleton_dataframe(self):
        """Return the growth tree as a dataframe (nodeid, parent, x, y, distance)."""
        return tree_to_dataframe(
            self.path_tree,
            name_col="nodeid_tree", parent_col="parent_id", path_col="path",
            attr_dict={"x": "x", "y": "y", "distance": "distance"},
        )
