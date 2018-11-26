import numpy as np
import scipy as sp
import more_itertools as mit

import part2
import part3

"""
We refer to the paper here: http://www.arnaud.martin.free.fr/publi/PARK_14a.pdf
"""

def generate_k(item):
    """k of deleted interpolation"""
    return (np.log(item + 1) + 1) / (np.log(item + 1) + 2)

def estimate_second_order_transitions(sequence):
    """
    As test data might not have Sj|Sj-1,Sj-2 due to data sparsity,
    we need to address this somehow.
    We can implement some sort of guessing based on the nearest state transitions,
    and will be using deleted interpolation for it.
    """

    A1, Y1, Z1, A1_frequency, Y1_frequency, Z1_frequency = part3.estimate_transitions(sequence) # first order
    label_counts = part3.get_label_counts(sequence)

    # now, we calculate the probabilities of each transition.
    K = len(label_counts)
    label_values = list(label_counts.values())
    label_keys   = list(label_counts)
    A2_frequency = np.zeros((K, K, K)) # Si, Sj -> Sk

    y_frequency = np.zeros(K)      # START_TOKEN -> S1
    Y_frequency = np.zeros((K, K)) # START_TOKEN, S1 -> S2, i is S1, j is S2

    z_frequency = np.zeros(K)      # Sn -> STOP
    Z_frequency = np.zeros((K, K)) # Sn-1, Sn -> STOP, i = Sn-1, j is Sn

    # TODO: rsplit beforehand
    for sentence in sequence:
        # we group each sentence with Si, Sj -> Sk
        window = [list(filter(None, w)) for w in mit.windowed(sentence, n=3, step=2)]

        # handle START_TOKEN, S1 -> S2
        S1 = window[0][0].rsplit(" ", 1)[1]
        S2 = window[0][1].rsplit(" ", 1)[1]
        S1_index = label_keys.index(S1)
        Y_frequency[S1_index, label_keys.index(S2)] += 1
        y_frequency[S1_index] += 1

        # handle Si, Sj -> Sk
        for triple in window:
            Si = triple[0]
            # Possible combinations are (Si, Sj, Sk), (Si, Sj) as overlap = 2.
            # We only are concerned with (Si, Sj, Sk), and leave (Si, Sj) for END_TOKEN.
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
        Sj_index = label_keys.index(Sj)

        z_frequency[Sj_index] += 1
        Z_frequency[label_keys.index(Si), Sj_index] += 1

    # we calculate the Ks for deleted interpolation
    k_function = np.vectorize(generate_k)
    k2 = k_function(A1_frequency)
    k3 = k_function(A2_frequency)

    lambda_1 = k3
    lambda_2 = (1 - k3) * k2
    lambda_3 = (1 - k3) * (1 - k2)

    #calculate the probabilities of Y and Z
    Y = Y_frequency / y_frequency[:,None]
    Z = Z_frequency / z_frequency[:,None]
    # we sum up over k to get the total count for each A[i, j]
    ij_counts = np.sum(A2_frequency, axis=2)

    # A2[i, j, k] = float(A_frequency[i, j, k]) / ij_counts[i, j]
    # first term is (K, K, K), second term is (K, K), third term is K.
    # we sum it up such that Si, Sj -> Sk + Sj -> Sk + Sk across all Sk.
    A2 = np.sum([lambda_1 * (A2_frequency / ij_counts[:,:,None]), (lambda_2 * A1)[:,:,None], (lambda_3 * (A1_frequency / np.sum(A1_frequency)))])

    return A2, Y, Z


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

        A2, Y, Z = estimate_second_order_transitions(training_set, label_counts)