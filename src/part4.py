import numpy as np
import scipy as sp
import more_itertools as mit

import part2
import part3

"""
We refer to the paper here: http://www.arnaud.martin.free.fr/publi/PARK_14a.pdf
"""

def estimate_second_order_transitions(sequence):
    """
    As training might not have Sj|Sj-1,Sj-2 due to data sparsity, we need to address this
    """
    A1, Y1, Z1 = part3.estimate_transitions(sequence) # first order
    label_counts = part3.get_label_counts(sequence)

    # now, we calculate the probabilities of each transition.
    K = len(label_counts)
    label_values = list(label_counts.values())
    label_keys   = list(label_counts)
    A2 = np.zeros((K, K, K)) # Si, Sj -> Sk

    # we define Y2 to be START_TOKEN, S1 -> S2
    Y = np.zeros((K, K))
    # we define Z to be Sn-1, Sn -> STOP
    Z = np.zeros((K, K))

    # TODO: rsplit beforehand
    for sentence in sequence:
        # we group each sentence with Si, Sj -> Sk
        window = [list(filter(None, w)) for w in mit.windowed(sentence, n=3, step=2)]

        # handle START_TOKEN, S1 -> S2
        S1 = window[0][0].rsplit(" ", 1)[1]
        S2 = window[0][1].rsplit(" ", 1)[1]

        Y[label_keys.index(S1), label_keys.index(S2)] += 1

        # handle Si, Sj -> Sk
        for triple in window:
            Si = triple[0]
            if len(triple) == 3:
                Sj = triple[1]
                Sk = triple[2]
                _observation, label = Si.rsplit(" ", 1)
                _observation, next_label = Sj.rsplit(" ", 1)
                _observation, last_label = Sk.rsplit(" ", 1)

                Si_index = label_keys.index(label)
                Sj_index = label_keys.index(next_label)
                Sj_index = label_keys.index(last_label)
                # store the indexes of the transition from Si -> Sj
                A2[Si_index, Sj_index, Sk_index] += 1

        # handle Si, Sj -> END_TOKEN
        Si = window[-1][-2].rsplit(" ", 1)[1]
        Sj = window[-1][-1].rsplit(" ", 1)[1]
        Z[label_keys.index(Si), label_keys.index(Sj)] += 1

    # TODO: calculate the probabilities of Y and Z

    # we sum up over k to get the total count for each A[i, j]
    ij_counts = np.sum(A, axis=2)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                A[i, j, k] = float(A[i, j, k]) / ij_counts[i, j]


def viterbi_2(sentence, X, S, Y, Z, A, B):
    """
    """
    pass

if __name__ == "__main__":
    for locale in ["EN"]:

        DATA = open(f"./../data/{locale}/train")
        training_set = part2.prepare_data(DATA)
        _results, observations, label_counts, emission_counts = part2.estimate_emissions(
            training_set)

        TEST_DATA = open(f"./../data/{locale}/dev.in")
        testing_set = part2.prepare_data(TEST_DATA)
        # with the test data, we are able to smooth out the emissions.
        B = part2.smooth_emissions(
            testing_set, observations, label_counts, emission_counts)

        estimate_second_order_transitions(training_set, label_counts)