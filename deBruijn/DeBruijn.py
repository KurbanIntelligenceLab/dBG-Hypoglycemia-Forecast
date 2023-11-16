import networkx as nx

import utils.IterationUtils as Iter

from utils.PropertyNames import GraphProperties as Props


class DeBruijnGraph:
    def __init__(self, k: int, sequences: list):
        """
        Initializes a De Bruijn graph from a list of sequences.

        :param k: Edge tuple (subsequence) size of the de Bruijn graph. The node tuple size is (k - 1)
        :type k: int
        :param sequences: The list of sequences to generate the graph from.
        :type sequences: list
        :raises ValueError: If the list of sequences is empty.
        """
        self.k = k
        if sequences is None or len(sequences) == 0:
            raise ValueError('No sequences found!')
        self.sequences = sequences
        self.alphabet = self._init_alphabet()
        self.graph = self._generate_graph()

    def __str__(self):
        """
        :return: String representation of the class
        """
        return f"{self.__class__.__name__} with " \
               f"{self.graph.number_of_nodes()} nodes " \
               f"and {self.graph.number_of_edges()} edges"

    def _init_alphabet(self):
        """
        Initializes the alphabet used in the sequences.

        :return: A set of unique characters found in the sequences.
        :rtype: set
        """
        alphabet = set()
        for sequence in self.sequences:
            for character in sequence:
                alphabet.add(character)
        return alphabet

    def update_graph(self, new_sequences: list, c):
        """
        Update the De Bruijn graph with new sequences.

        :param new_sequences: The list of new sequences to update the graph.
        :type new_sequences: list
        :raises ValueError: If the list of new sequences is empty.
        """
        if not new_sequences:
            raise ValueError('No new sequences provided!')

        # Add new sequences to the existing sequences
        self.sequences.extend(new_sequences)

        # Update alphabet
        for sequence in new_sequences:
            for character in sequence:
                self.alphabet.add(character)

        # Update the existing graph with new sequences
        prev = None
        for sequence in new_sequences:
            for index, current in enumerate(Iter.sliding_window(sequence, self.k - 1)):
                if index == 0:
                    prev = current
                    continue
                # Increment the edge weight by 1 if it already exists in graph
                if self.graph.has_edge(prev, current):
                    self.graph.edges[prev, current][Props.weight] += 1
                else:
                    # Create an edge with 1 weight if it does not exist in the graph
                    edge_attributes = {
                        Props.weight: c * 1.0,
                        Props.tuple: (prev + current[-1:])
                    }
                    self.graph.add_edge(prev, current, **edge_attributes)
                prev = current

    def _generate_graph(self):
        """
        Generates the De Bruijn graph from the sequences.

        :return: A networkx DiGraph representing the De Bruijn graph.
        :rtype: networkx.classes.digraph.DiGraph
        """
        debruijn_graph = nx.DiGraph()
        prev = None
        for sequence in self.sequences:
            for index, current in enumerate(Iter.sliding_window(sequence, self.k - 1)):
                if index == 0:
                    prev = current
                    continue
                # Increment the edge weight by 1 if it already exists in graph
                if debruijn_graph.has_edge(prev, current):
                    debruijn_graph.edges[prev, current][Props.weight] += 1
                else:
                    # Create an edge with 1 weight if it does not exist in the graph
                    edge_attributes = {
                        Props.weight: 1.0,
                        Props.tuple: (prev + current[-1:])
                    }
                    debruijn_graph.add_edge(prev, current, **edge_attributes)
                prev = current
        return debruijn_graph
