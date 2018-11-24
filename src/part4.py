import numpy as np
import part2

def viterbi(sentence, X, S, Y, A, B):
    """
    This algorithm generates a path which is a sequence of labels that generates the observations.
    The code references the pseudocode from Wikipedia https://en.wikipedia.org/wiki/Viterbi_algorithm.
    It is modified to suit our previous codes.

    We define these symbols:
    X => the set of observations. There are N unique observations, and T observations in this particular sentence.
    S => the set of labels. There are K unique labels, and T labels in this particular sentence.

    other inputs are:
    sentence => the sentence we are analyzing. it is T-long.
    Y => list of initial transistion probabilities, such that Yi stores the probability that Y1 == Si.
    A => transition matrix of size K x K such that Aij stores the transition probability of transiting from si to sj.
    B => emission matrix of size K x N such that Bij stores the probability of observing oj from si.
    """

    K = len(S)
    T = len(sentence)

    T1 = np.zeros((K, T))  # T1[i,j] stores the probability of the most likely path so far
    T2 = np.zeros((K, T))  # T2[i,j] stores the parent of the most likely path so far
    result = np.zeroes(T)

    # base case
    for i in range(K):
        T1[i, 0] = Y[i]*B[i, 0]
        # T2[i, 0] = 0

    # recursive case
    for i in range(1, T):
        for j in range(K):
            calc = [T1[k, i-1] * A[k, j] * B[j, i] for k in range(K)]
            max_value = np.amax(calc)
            # find the maximum value and store into T1. store the k responsible into T2.
            T1[j, i] = max_value
            T2[j, i] = calc.index(max_value)

    # end case
    # we have a list to store the largest values. we go through T2 to obtain back the best path.
    Z = np.zeroes(T)

    last_values = [T1[k, T-1] for k in range(K)]
    Z[T-1] = last_values.index(np.amax(last_values)) # find the k index responsible for largest value.
    result[T-1] = S[Z[T-1]] # get the optimal label by index.

    for i in range(T-1, 0, -1): # from the 2nd last item to the first item.
        Z[i-1] = T2[Z[i], i]
        result[i-1] = S[Z[i-1]]

    return result

def viterbi_2(sentence, X, S, Y, A, B):
    """
    This algorithm is similar to the original viterbi function except with second order transition probabilities.
    """

    K = len(S)
    T = len(sentence)

    T1 = np.zeros((K, T))  # T1[i,j] stores the probability of the most likely path so far
    T2 = np.zeros((K, T))  # T2[i,j] stores the parent of the most likely path so far
    result = np.zeroes(T)

    # base case remains the same as the original function
    for i in range(K):
        T1[i, 0] = Y[i]*B[i, 0]
        # T2[i, 0] = 0

    # new case before the actual recursive case, at index = 1. Same as the original viterbi's recursive case,
    # as it has only one parent.
    for i in range(K):
        calc = [T1[k, i-1] * A[k, i] * B[i, 1] for k in range(K)]
        max_value = np.amax(calc)
        # find the maximum value and store into T1. store the k responsible into T2.
        T1[i, 1] = max_value
        T2[i, 1] = calc.index(max_value)

    # recursive case. Same as the original viterbi's recursive case except with an extra T1 with i-2.
    for i in range(2, T):
        for j in range(K):
            calc = [T1[k, i-2] * T1[k, i-1] * A[k, j] * B[j, i] for k in range(K)]
            max_value = np.amax(calc)
            # find the maximum value and store into T1. store the k responsible into T2.
            T1[j, i] = max_value
            T2[j, i] = calc.index(max_value)

    # end case
    # we have a list to store the largest values. we go through T2 to obtain back the best path.
    # the backward propagation is the same as the original viterbi algorithm.
    Z = np.zeroes(T)

    last_values = [T1[k, T-1] for k in range(K)]
    Z[T-1] = last_values.index(np.amax(last_values)) # find the k index responsible for largest value.
    result[T-1] = S[Z[T-1]] # get the optimal label by index.

    for i in range(T-1, 0, -1): # from the 2nd last item to the first item.
        Z[i-1] = T2[Z[i], i]
        result[i-1] = S[Z[i-1]]

    return result

if __name__ == "__main__":
    for locale in ["EN", "FR"]:
        DATA = open(f"./../data/{locale}/train")
        training_set = part2.prepare_data(DATA)
        _results, observations, label_counts, emission_counts = part2.estimate_emissions(
            training_set)

        TEST_DATA = open(f"./../data/{locale}/dev.in")
        testing_set = part2.prepare_data(TEST_DATA)
        # with the test data, we are able to smooth out the emissions.
        results = part2.smooth_emissions(
            testing_set, observations, label_counts, emission_counts)

        # from the results, we generate a K x N matrix.
        B = np.zeros((len(label_counts), len(results)))
        for j, (j_key, j_value) in enumerate(results.items()):
            for i, i_key in enumerate(label_counts.keys()):
                if i_key in j_value:
                    # if the label emits the observation at j
                    B[i, j] = j_value[i_key]
        print(f"S -> {label_counts.keys()}")
        print(f"X -> {len(results)} observations")
        print(B.shape)