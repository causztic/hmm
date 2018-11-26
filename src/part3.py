import numpy as np
import more_itertools as mit
import part2

def get_label_counts(sequence):
    """
    Obtain the counts of every label.
    sequence : a list of sentences from the training set.
    """

    label_counts = {}

    # we do a first-pass to obtain the counts of every label.
    for sentence in sequence:
        for item in sentence:
            _observation, label = item.rsplit(" ", 1)
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

    return label_counts

def estimate_transitions(sequence):
    """
    Estimates the transition parameters from the training set using MLE.
    sequence : a list of sentences from the training set.
    """
    label_counts = get_label_counts(sequence)

    # now, we calculate the probabilities of each transition.
    K = len(label_counts)
    label_values = np.array(list(label_counts.values()))
    label_keys   = list(label_counts)
    A_frequency = np.zeros((K, K))

    # we define Y to be START_TOKEN -> S1
    Y_frequency = np.zeros(K)
    # we define Z to be Sn -> END_TOKEN
    Z_frequency = np.zeros(K)

    for sentence in sequence:
        # we group each sentence with Si -> Sj
        window = [list(filter(None, w)) for w in mit.windowed(sentence, n=2, step=1)]

        # handle START_TOKEN -> S1
        Y_frequency[label_keys.index(window[0][0].rsplit(" ", 1)[1])] += 1

        # handle Si -> Sj
        for pair in window:
            Si = pair[0]
            if len(pair) == 2:
                Sj = pair[1]
                # more than one word
                _observation, label = Si.rsplit(" ", 1)
                _observation, next_label = Sj.rsplit(" ", 1)
                Si_index = label_keys.index(label)
                Sj_index = label_keys.index(next_label)
                # store the indexes of the transition from Si -> Sj
                A_frequency[Si_index, Sj_index] += 1

        # handle Sn -> END_TOKEN
        Z_frequency[label_keys.index(window[-1][-1].rsplit(" ", 1)[1])] += 1

    # calculate the probabilities of START_TOKEN -> S1 and Sn -> END_TOKEN
    Y = Y_frequency / len(sequence)
    Z = Z_frequency / len(sequence)

    # now that results have all the counts of transitions,
    # we use the label_counts to divide them to get the MLE
    A = A_frequency / label_values[:,None]

    return A, Y, Z, A_frequency, Y_frequency, Z_frequency


"""TODO: in case you encounter potential numerical underflow issue, think of a way to address such an
issue in your implementation.
"""


def viterbi(sentence, X, S, Y, Z, A, B):
    """
    This algorithm generates a path which is a sequence of labels that generates the observations.
    The code references the pseudocode from Wikipedia https://en.wikipedia.org/wiki/Viterbi_algorithm.
    It is modified to suit our previous codes.

    Parameters
    sentence => the sentence we are analyzing. it is T-long.
    X => list of observations. There are N unique observations, and T observations in this particular sentence.
    S => list of labels. There are K unique labels (excluding START_TOKEN AND STOP_TOKEN), and T labels in this particular sentence.
    Y => list of probabilities of START_TOKEN -> S1
    Z => list of probabilities of Sn -> END_TOKEN
    A => transition matrix of size K x K such that Aij stores the transition probability of transiting from Si to Sj.
    B => emission matrix of size K x N such that Bij stores the probability of observing oj from Si.
    """

    K = len(S)
    T = len(sentence)

    # T1[i,j] stores the probability of the most likely path so far
    T1 = np.zeros((K, T+1))
    # T2[i,j] stores the parent of the most likely path so far
    T2 = np.zeros((K, T+1))
    result = []

    # first case, START -> S1
    idx = -1 # UNKNOWN_TOKEN
    if sentence[0] in X:
        idx = X.index(sentence[0])
    for i in range(K):
        # T2[i, 0] = 0
        T1[i, 0] = np.log(Y[i]) + np.log(B[i, idx])
    # recursive case
    for i in range(1, T):
        idx = -1
        if sentence[i] in X:
            idx = X.index(sentence[i])
        for j in range(K):
            calc = [T1[k, i-1] + np.log(A[k, j]) + np.log(B[j, idx]) for k in range(K)]

            max_index = np.argmax(calc)
            # find the maximum value and store into T1. store the k responsible into T2.
            T1[j, i] = calc[max_index]
            T2[j, i] = max_index
    # end case
    # we omit B as STOP will not have a B value (all 0)
    for i in range(K):
        calc = [T1[k, T-1] + np.log(Z[i]) for k in range(K)]
        max_index = np.argmax(calc)
        # find the maximum value and store into T1. store the k responsible into T2.
        T1[i, T] = calc[max_index]
        T2[i, T] = max_index

    # we have a list to store the largest values. we go through T2 to obtain back the best path.
    W = np.zeros(T+1, dtype=np.int8)

    # find the k index responsible for largest value.
    W[T] = np.argmax(T1[:,T])
    result.append(S[W[T]])  # get the optimal label by index.

    for i in range(T, 0, -1):  # from the 2nd last item to the first item.
        W[i-1] = T2[W[i], i]
        result.append(S[W[i-1]])
    result.reverse()
    return result

def predict_viterbi(locale, observations, labels, Y, Z, A, B):
    """Get most probable label -> observation with Viterbi, and write to file."""

    testing_set = [line.rstrip("\n")
                    for line in open(f"./../data/{locale}/dev.in")]

    file = open(f"./../data/{locale}/dev.p3.out", "w")
    sentence_buffer = []
    count = 0
    for line in testing_set:
        if not line.strip():
            # sentence has ended
            result = viterbi(sentence_buffer, observations, labels, Y, Z, A, B)
            for index, word in enumerate(sentence_buffer):
                file.write(f"{word} {result[index]}\n")
            file.write("\n")
            sentence_buffer = []
        else:
            sentence_buffer.append(line)
    file.close()

if __name__ == "__main__":
    for locale in ["EN", "FR", "CN", "SG"]:
        print(f"Running for {locale} dataset")
        DATA = open(f"./../data/{locale}/train")
        training_set = part2.prepare_data(DATA)
        _results, observations, label_counts, emission_counts = part2.estimate_emissions(
            training_set)

        TEST_DATA = open(f"./../data/{locale}/dev.in")
        testing_set = part2.prepare_data(TEST_DATA)
        # with the test data, we are able to smooth out the emissions.
        B = part2.smooth_emissions(
            testing_set, observations, label_counts, emission_counts)

        A, Y, Z, _, _, _ = estimate_transitions(training_set)

        predict_viterbi(locale, observations, list(label_counts), Y, Z, A, B)