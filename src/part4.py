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

    Y_frequency = np.zeros((K, K)) # START_TOKEN, Si -> Sj
    Z_frequency = np.zeros((K, K)) # Si, Sj -> STOP

    for sentence in sequence:
        labels_for_sentence = [word.rsplit(" ", 1)[1] for word in sentence]
        # we group each sentence with Si, Sj -> Sk
        # window = [item.rsplit(" ", 1)[1] for item in list(filter(None, w) for w in mit.windowed(sentence, n=3, step=2))]
        window = [list(filter(None, w)) for w in mit.windowed(labels_for_sentence, n=3, step=1)]
        # handle START_TOKEN, S1 -> S2
        S1 = window[0][0]
        if len(window[0]) > 1:
            S2 = window[0][1]
            S1_index = label_keys.index(S1)
            Y_frequency[S1_index, label_keys.index(S2)] += 1

            # handle Si, Sj -> Sk
            for triple in window:
                Si = triple[0]
                # Possible combinations are (Si, Sj, Sk), (Si, Sj) as overlap = 2.
                # We only are concerned with (Si, Sj, Sk), and leave (Si, Sj) for END_TOKEN.
                if len(triple) == 3:
                    Sj = triple[1]
                    Sk = triple[2]

                    Si_index = label_keys.index(Si)
                    Sj_index = label_keys.index(Sj)
                    Sk_index = label_keys.index(Sk)
                    # store the indexes of the transition from Si -> Sj
                    A2_frequency[Si_index, Sj_index, Sk_index] += 1


        # handle Sj -> END_TOKEN
        Sj = window[-1][-1]
        Sj_index = label_keys.index(Sj)
        # handle Si, Sj -> END_TOKEN
        if len(window[0]) > 1:
            Si = window[-1][-2]
            Z_frequency[label_keys.index(Si), Sj_index] += 1

    # we calculate the Ks for deleted interpolation
    k_function = np.vectorize(generate_k)
    k2 = k_function(A1_frequency)
    k3 = k_function(A2_frequency)

    lambda_1 = k3
    lambda_2 = (1 - k3) * k2[:,None]
    lambda_3 = (1 - k3) * (1 - k2[:,None])

    #calculate the probabilities of Y and Z
    Y = np.nan_to_num(Y_frequency / Y1_frequency[:,None]) # convert to 0 if the specific frequency is 0
    Z = np.nan_to_num(Z_frequency / Z1_frequency[:,None])
    # we sum up over k to get the total count for each A[i, j]
    ij_counts = np.sum(A2_frequency, axis=2)

    # A2[i, j, k] = float(A_frequency[i, j, k]) / ij_counts[i, j]
    # first term is (K, K, K), second term is (K, K), third term is K.
    # we sum it up such that Si, Sj -> Sk + Sj -> Sk + Sk across all Sk.
    second_order = lambda_1 * np.nan_to_num(A2_frequency / ij_counts[:,:,None])
    first_order  = lambda_2 * A1
    zero_order   = lambda_3 * np.nan_to_num(A1_frequency / np.sum(A1_frequency))
    A2 = np.sum([second_order, first_order, zero_order], axis=0)
    return A2, Y, Y1, Z, Z1


def viterbi_2(sentence, X, S, Y, Y1, Z, Z1, A, B):
    """
    Modified Viterbi implementation to allow second order transitions.
    In addition to the variables in viterbi(), we introduce:

    Y1 => START_TOKEN -> S1
    Y  => START_TOKEN, S1 -> S2
    Z  => Sn-1, Sn -> END_TOKEN
    Z1 => Sn -> END_TOKEN
    """

    K = len(S)
    N = len(sentence)

    # T1[i,j] stores the probability of the most likely path so far
    T1 = np.zeros((K, N+1))
    # T2[i,j] stores the parent of the most likely path so far
    T2 = np.zeros((K, N+1))
    result = []

    for n in range(0,N):
        idx = -1
        if sentence[n] in X:
            idx = X.index(sentence[n])

        if n == 0:
            # base cases, START -> Sj
            for k in range(K):
                T1[k, 0] = np.log(Y1[k]) + np.log(B[k, idx])
        elif n == 1:
            # START, Sj -> Sk
            for k in range(K):
                calc = [T1[j, 1] + np.log(Y[j, k]) + np.log(B[k, idx]) for j in range(K)]
                max_index = np.argmax(calc)
                T1[k, n] = calc[max_index]
                T2[k, n] = max_index
        else:
            # recursive case, Si, Sj -> Sk
            for k in range(K):
                calc = []
                for j in range(K):
                    for i in range(K):
                        calc.append(T1[j, n-1] + np.log(A[i, j, k]) + np.log(B[k, idx]))

                max_index = np.argmax(calc)
                T1[k, n] = calc[max_index]
                T2[k, n] = np.unravel_index(max_index, (K, K))[1]

    # end case, Sn-1 Sn -> END
    for j in range(K):
        calc = [T1[j, N-1] + np.log(Z[i, j]) for i in range(K)]

        max_index = np.argmax(calc)
        # find the maximum value and store into T1. store the k responsible into T2.
        T1[j, N] = calc[max_index]
        T2[j, N] = max_index

    W = np.zeros(N+1, dtype=np.int8)
    # find the k index responsible for largest Sn -> STOP value.
    W[N] = np.argmax(T1[:,N])
    for i in range(N, 0, -1):  # from the 2nd last item to the first item.
        W[i-1] = T2[W[i], i]
        result.append(S[W[i-1]])
    result.reverse()
    return result


def predict_viterbi_2(locale, observations, labels, Y, Y1, Z, Z1, A, B):
    """Get most probable label -> observation with second-order Viterbi, and write to file."""

    testing_set = [line.rstrip("\n")
                    for line in open(f"./../data/{locale}/dev.short.in")]

    file = open(f"./../data/{locale}/dev.p4.out", "w")
    sentence_buffer = []
    for index, line in enumerate(testing_set):
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