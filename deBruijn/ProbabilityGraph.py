from networkx.classes.digraph import DiGraph

from deBruijn.DeBruijn import DeBruijnGraph
from models.ProbabilisticModel import ProbabilisticModel
from utils.PropertyNames import GraphProperties as Props
from utils.PropertyNames import MethodOptions as Opts


class ProbabilityGraph(DeBruijnGraph):

    def __init__(self, k: int, sequences: list, risky_chars: set = None):
        """
        Initializes a ProbabilityGraph which is an extended de Bruijn graph

        :param k: Edge tuple (subsequence) size of the de Bruijn graph. The node tuple size is (k - 1)
        :type k: int

        :param sequences: The list of sequences to generate the graph from.
        :type sequences: list

        :param risky_chars: The set of characters in the dataset that are considered dangerous. Takes the top 2 and
        bottom 2 chars as default.

        :type risky_chars: set
        """
        super().__init__(k, sequences)
        if risky_chars is None:
            smallest = sorted(set(self.alphabet))[:2]
            largest = sorted(set(self.alphabet), reverse=True)[:2]
            self.risky_chars = set(smallest + largest)
        else:
            self.risky_chars = risky_chars

    def get_probability_model(self,
                              risk_threshold: float = 0.5,
                              prune_method: str = None,
                              prune_threshold: float = None,
                              max_steps: int = 3,
                              prune: bool = False):
        """
        Generates a probability table form the de Bruijn graph. The probability of a node represents the likelihood of
        ending up in a 'dangerous node' after taking `max_steps` steps in the graph.

        :param risk_threshold: Used to determine sensitivity of the alert. We give alert if the risk probability is
        bigger than this threshold.
        :type risk_threshold: float

        :param prune_method: The method used for pruning the graph. Either 'path' or 'filter'. Path pruning starts from
        the 'dangerous nodes' and traverses backwards from those nodes until we hit `prune_threshold`. Ignored if
        `prune` is False.
        :type prune_method: str

        :param prune_threshold: The threshold for pruning the graph. The bigger values will result in fewer edges.
        Ignored if `prune` is False.
        :type prune_threshold: float

        :param max_steps: The maximum number of steps to take into account while calculating the probabilities
        :type max_steps: int

        :param prune: A boolean indicating whether to prune the graph.
        :type prune: bool

        :return: ProbabilisticModel from the de Bruijn graph
        :rtype: ProbabilisticModel

        :raises ValueError: If the prune method is neither 'path' nor 'filter'.
        """

        # Prune graph
        if prune:
            if prune_method == Opts.path:
                subgraph = self._path_prune(prune_threshold)
            elif prune_method == Opts.filter:
                subgraph = self._filter_prune(prune_threshold)
            else:
                raise ValueError('Unknown prune method!')
        else:
            subgraph = self.graph

        # Find dangerous nodes by finding nodes that contain risky chars
        dangerous_nodes = set()
        for node in subgraph.nodes:
            if any(element in node for element in self.risky_chars):
                dangerous_nodes.add(node)

        # Fill dictionary with probability values
        prob_dict = dict()
        for node in subgraph.nodes:
            prob_dict[node] = self._get_probability(subgraph, node, dangerous_nodes, max_steps)

        # draw_hist(list(prob_dict.values()), max_steps, title='Distribution of Probability of Leading Into a Risky Area in {} Steps'.format(title))

        return ProbabilisticModel(self.k, prob_dict, risk_threshold, self.risky_chars)

    def _get_probability(self, subgraph: DiGraph, start_node: tuple, target_nodes: set, step_count: int):
        """
        Calculates the probability of reaching a risky node from the given start node within the given number of steps.
        On branching paths, treats edge weights as a weighted probability of choosing that edge.

        :param subgraph: The subgraph to calculate the probability on.
        :type subgraph: networkx.classes.digraph.DiGraph

        :param start_node: The node to start the calculation from.
        :type start_node: tuple

        :param target_nodes: The set of nodes considered dangerous.
        :type target_nodes: set

        :param step_count: The maximum number of steps to consider.
        :type step_count: int

        :return: The calculated probability.
        :rtype: float
        """
        stack = [(start_node, 0, 1)]
        probability = 0

        while stack:
            node, depth, prob = stack.pop()

            # Check if the current node is a target node
            if node in target_nodes:
                probability += prob
                continue

            # Check if we reached the maximum number of steps
            if depth >= step_count:
                continue

            # Get total weight of all outgoing edges
            total_weight = sum(edge[2][Props.weight] for edge in self.graph.edges(node, data=True))

            # Add all neighbors to the stack
            for neighbor, edge in zip(subgraph.neighbors(node), subgraph.edges(node, data=True)):
                stack.append((neighbor, depth + 1, prob * edge[2][Props.weight] / total_weight))

        return probability

    def _filter_prune(self, threshold: float):
        """
        Prunes the graph by only including nodes connected by edges with weight above the given threshold.

        :param threshold: The weight threshold for including nodes.
        :type threshold: float

        :return: The pruned subgraph.
        :rtype: networkx.classes.digraph.DiGraph
        """
        included_nodes = set()
        for edge in self.graph.edges:
            network_edge = self.graph.edges[edge]
            if network_edge[Props.weight] >= threshold:
                included_nodes.add(edge[0])
                included_nodes.add(edge[1])
        subgraph = self.graph.subgraph(included_nodes).copy()
        return subgraph

    def _path_prune(self, threshold: float):
        """
        Prunes the graph by only including nodes reachable by traversing edges with weight above the given threshold.

        :param threshold: The weight threshold for including nodes.
        :type threshold: float

        :return: The pruned subgraph.
        :rtype: networkx.classes.digraph.DiGraph
        """
        traversed_nodes = set()
        for edge in self.graph.edges:
            network_edge = self.graph.edges[edge]
            # Do backward traversal if the node is a dangerous node and collect every discovered node in
            # `traversed_nodes`
            if any(element in network_edge[Props.tuple] for element in self.risky_chars) \
                    and network_edge[Props.weight] >= threshold:
                traversed_nodes.add(edge[1])
                traversed = self._backward_traversal(edge, threshold)
                traversed_nodes.update(traversed)
        subgraph = self.graph.subgraph(traversed_nodes).copy()
        return subgraph

    def _backward_traversal(self, start_edge: tuple, threshold: float):
        """
        Traverses the graph backwards from the given edge, including nodes connected by edges with weight above the
        given threshold.

        :param start_edge: The edge to start the traversal from.
        :type start_edge: tuple

        :param threshold: The weight threshold for including nodes.
        :type threshold: float

        :return: The set of traversed nodes.
        :rtype: set
        """
        traversed_nodes = set()
        stack = [start_edge[0]]
        while stack:
            node = stack.pop()
            traversed_nodes.add(node)
            untraversed_incoming_nodes = [(in_edge[0], in_edge[2][Props.weight]) for in_edge in
                                          self.graph.in_edges(node, data=True) if
                                          in_edge[0] not in traversed_nodes]
            for incoming_edge in untraversed_incoming_nodes:
                if incoming_edge[1] > threshold:
                    stack.append(incoming_edge[0])
        return traversed_nodes
