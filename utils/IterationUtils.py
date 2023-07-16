"""
This file contains helper iteration functions
"""

from itertools import islice


def sliding_window(sequence, window_size: int):
    """
    Generate a sliding window over a given sequence.

    This function takes a sequence (like a list or a string) and a window size,
    and generates a new sequence of 'windows', where each window is a tuple of `window_size`
    consecutive elements from the original sequence.

    :param sequence: The sequence to generate the sliding window over. This should be an
                     iterable like a list or a string.
    :type sequence: iterable
    :param window_size: The size of the sliding window. This should be a positive integer.
    :type window_size: int
    :return: An iterator that yields tuples of `window_size` consecutive elements from `sequence`.
    :rtype: iterator
    """
    it = iter(sequence)

    result = tuple(islice(it, window_size))
    if len(result) == window_size:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def loo_partition(sequences: list):
    """
    Partition a list of sequences for Leave-One-Out (LOO) cross-validation.

    This function takes a list of sequences and generates pairs of training and test data.
    For each iteration, it selects one sequence for testing and the remaining sequences for training.

    :param sequences: The list of sequences to be partitioned. Each sequence can be any iterable object.
    :type sequences: list
    :return: An iterator that yields pairs of training and test data. The training data is a list of
             sequences, and the test data is a single sequence.
    :rtype: iterator
    """

    for i in range(len(sequences)):
        train_data = list(sequences)
        test_data = train_data.pop(i)

        yield train_data, test_data
