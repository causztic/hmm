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
        if len(window[0]) > 1:
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
                    Sk_index = label_keys.index(last_label)
                    # store the indexes of the transition from Si -> Sj
                    A2_frequency[Si_index, Sj_index, Sk_index] += 1


        # handle Sj -> END_TOKEN
        Sj = window[-1][-1].rsplit(" ", 1)[1]
        Sj_index = label_keys.index(Sj)
        z_frequency[Sj_index] += 1
        # handle Si, Sj -> END_TOKEN
        if len(window[0]) > 1:
            Si = window[-1][-2].rsplit(" ", 1)[1]
            Z_frequency[label_keys.index(Si), Sj_index] += 1

    # we calculate the Ks for deleted interpolation
    k_function = np.vectorize(generate_k)
    k2 = k_function(A1_frequency)
    k3 = k_function(A2_frequency)

    lambda_1 = k3
    lambda_2 = (1 - k3) * k2[:,None]
    lambda_3 = (1 - k3) * (1 - k2[:,None])

    #calculate the probabilities of Y and Z
    Y = np.nan_to_num(Y_frequency / y_frequency[:,None]) # convert to 0 if the specific frequency is 0
    Z = np.nan_to_num(Z_frequency / z_frequency[:,None])
    # we sum up over k to get the total count for each A[i, j]
    ij_counts = np.sum(A2_frequency, axis=2)

    # A2[i, j, k] = float(A_frequency[i, j, k]) / ij_counts[i, j]
    # first term is (K, K, K), second term is (K, K), third term is K.
    # we sum it up such that Si, Sj -> Sk + Sj -> Sk + Sk across all Sk.
    second_order = lambda_1 * (A2_frequency / ij_counts[:,:,None])
    first_order  = lambda_2 * A1
    zero_order   = lambda_3 * (A1_frequency / np.sum(A1_frequency))
    A2 = np.nan_to_num(np.sum([second_order, first_order, zero_order], axis=0))

    return A2, Y, Y1, Z, Z1


def viterbi_2(sentence, X, S, Y, Y1, Z, Z1, A, B):
    """
    Modified Viterbi implementation to allow second order transitions.
    In addition to the variables in viterbi(), we introduce:

    Y1 => START_TOKEN -> S1
    Y  => START_TOKEN, S1 -> S2
    Z  => Sn-1, Sn -> END_TOKEN
    """

    K = len(S)
    T = len(sentence)

    # T1[i,j, k] stores the probability of the most likely path so far, where Si, Sj -> Sk
    T1 = np.zeros((K, K, T+1))
    # T2[i,j] stores the parent of the most likely path so far
    T2 = np.zeros((K, K, T+1))
    result = []

    # first case, START -> S1
    # T2[i, 0] = 0
    idx = -1 # UNKNOWN_TOKEN
    if sentence[0] in X:
        idx = X.index(sentence[0])
    for j in range(K):
        T1[:, j, 0] = np.log(Y1[j]) + np.log(B[j, idx])

    # second case, START, S1 -> S2
    idx = -1 # UNKNOWN_TOKEN
    if sentence[1] in X:
        idx  = X.index(sentence[1])
    for i in range(K):
        for j in range(K):
            # T2[i, 0] = 0
            T1[i, j, 1] = np.log(Y[i, j]) + np.log(B[j, idx])

    # recursive case, Si, Sj -> Sk
    for o in range(2, T):
        idx = -1
        if sentence[o] in X:
            idx = X.index(sentence[o])
        for i in range(K):
            for j in range(K):
                calc = [T1[k, l, o-1] + np.log(A[k, l, j]) + np.log(B[j, idx]) for k in range(K) for l in range(K)]

                max_index = np.argmax(calc)
                # find the maximum value and store into T1. store the k responsible into T2.
                T1[i, j, o] = calc[max_index]
                T2[i, j, o] = max_index

    # last case Sn-1, Sn -> END
    # we omit B as STOP will not have a B value (all 0)

    if T > 1:
        for i in range(K):
            for j in range(K):
                calc = [T1[k, l, T-1] + np.log(Z[i, j]) for k in range(K) for l in range(K)]
                max_index = np.argmax(calc)
                # find the maximum value and store into T1. store the k responsible into T2.
                T1[i, j, T] = calc[max_index]
                T2[i, j, T] = max_index
    else:
        # single word prediction
        for j in range(K):
            calc = [T1[k, l, T-1] + np.log(Z1[i]) for k in range(K) for l in range(K)]
            max_index = np.argmax(calc)
            # find the maximum value and store into T1. store the k responsible into T2.
            T1[:, j, T] = calc[max_index]
            T2[:, j, T] = max_index


    # we have a list to store the largest values. we go through T2 to obtain back the best path.
    W = np.zeros(T+1, dtype=np.int8)

    # find the k index responsible for largest value.
    print(np.argmax(T1[:, :, T]), np.argmax(T1[:, :, T], axis=0))
    W[T] = np.argmax(T1[:, :, T])
    result.append(S[W[T]])  # get the optimal label by index.

    for i in range(T, 0, -1):  # from the 2nd last item to the first item.
        W[i-1] = T2[W[i], i]
        result.append(S[W[i-1]])
    result.reverse()
    return result

def predict_viterbi_2(locale, observations, labels, Y, Y1, Z, Z1, A, B):
    """Get most probable label -> observation with second-order Viterbi, and write to file."""

    testing_set = [line.rstrip("\n")
                    for line in open(f"./../data/{locale}/dev.in")]

    file = open(f"./../data/{locale}/dev.p4.out", "w")
    sentence_buffer = []
    for line in testing_set:
        if not line.strip():
            # sentence has ended
            result = viterbi_2(sentence_buffer, observations, labels, Y, Y1, Z, Z1, A, B)
            for index, word in enumerate(sentence_buffer):
                file.write(f"{word} {result[index]}\n")
            file.write("\n")
            sentence_buffer = []
        else:
            sentence_buffer.append(line)
    file.close()

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

        A, Y, Y1, Z, Z1 = estimate_second_order_transitions(training_set)

        predict_viterbi_2(locale, observations, list(label_counts), Y, Y1, Z, Z1, A, B)