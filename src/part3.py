import numpy as np
import more_itertools as mit
import part2

START_TOKEN = "#START#"
STOP_TOKEN = "#STOP#"


def estimate_transitions(sequence, group_size = 2, overlap = 1):
    """
    Estimates the transition parameters from the training set using MLE.
    group_size is the order of the transitions + 1
    overlap is the order of the transitions.
    """
    label_counts = {START_TOKEN: 0}  # count of every label

    # we do a first-pass to obtain the counts of every label.
    for sentence in sequence:
        # add to label_counts
        label_counts[START_TOKEN] += 1

        for item in sentence:
            _observation, label = item.rsplit(" ", 1)
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

    # we add STOP_TOKEN at the end to preserve state order.
    label_counts[STOP_TOKEN] = len(sequence)


    # now, we calculate the probabilities of each transition.
    K = len(label_counts)
    A = np.zeros((K, K))

    for sentence in sequence:
        # we group each sentence with Si -> Sj
        window = list(mit.windowed(sentence, n=group_size, step=overlap))
        # handle START_TOKEN -> S1
        A[0, label_counts.keys().index(window[0].rsplit(" ", 1)[1])] += 1

        # handle Si -> Sj
        for pair in window:
            Si, Sj = pair
            _observation, label = Si.rsplit(" ", 1)
            _observation, next_label = Sj.rsplit(" ", 1)
            Si_index = label_counts.keys().index(label)
            Sj_index = label_counts.keys().index(next_label)
            # store the indexes of the transition from Si -> Sj
            A[Si_index, Sj_index] += 1

        # handle Sn -> END_TOKEN
        A[len(label_counts) - 1, label_counts.keys().index(window[-1].rsplit(" ", 1)[1])] += 1


    # now that results have all the counts of transitions,
    # we use the label_counts to divide them to get the MLE
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = float(A[i, j]) / label_counts.values()[i]

    return label_counts, A


"""TODO: in case you encounter potential numerical underflow issue, think of a way to address such an
issue in your implementation.
"""


def viterbi(sentence, X, S, A, B):
    """
    This algorithm generates a path which is a sequence of labels that generates the observations.
    The code references the pseudocode from Wikipedia https://en.wikipedia.org/wiki/Viterbi_algorithm.
    It is modified to suit our previous codes.

    We define these symbols:
    X => the set of observations. There are N unique observations, and T observations in this particular sentence.
    S => the set of labels. There are K unique labels, and T labels in this particular sentence.

    other inputs are:
    sentence => the sentence we are analyzing. it is T-long.
    A => transition matrix of size K x K such that Aij stores the transition probability of transiting from si to sj.
    B => emission matrix of size K x N such that Bij stores the probability of observing oj from si.
    """

    K = len(S)
    T = len(sentence)

    # T1[i,j] stores the probability of the most likely path so far
    T1 = np.zeros((K, T))
    # T2[i,j] stores the parent of the most likely path so far
    T2 = np.zeros((K, T))
    result = np.zeroes(T)

    # first case, START -> S1
    for i in range(K):
        T1[i, 0] = A[0, i]*B[i, 0]
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
    # we omit B as STOP will not have a B value (all 0)
    for j in range(K):
        calc = [T1[k, T-1] * A[k, j] for k in range(K)]
        max_value = np.amax(calc)
        # find the maximum value and store into T1. store the k responsible into T2.
        T1[j, i] = max_value
        T2[j, i] = calc.index(max_value)


    # we have a list to store the largest values. we go through T2 to obtain back the best path.
    Z = np.zeroes(T)

    last_values = [T1[k, T-1] for k in range(K)]
    # find the k index responsible for largest value.
    Z[T-1] = last_values.index(np.amax(last_values))
    result[T-1] = S[Z[T-1]]  # get the optimal label by index.

    for i in range(T-1, 0, -1):  # from the 2nd last item to the first item.
        Z[i-1] = T2[Z[i], i]
        result[i-1] = S[Z[i-1]]

    return result

def predict_viterbi(locale, observations, labels, A, B):
    """Get most probable label -> observation with Viterbi, and write to file."""

    training_set = [line.rstrip("\n")
                    for line in open(f"./../data/{locale}/dev.in")]

    file = open(f"./../data/{locale}/dev.p3.out", "w")
    sentence_buffer = []
    viterbi()
    for line in training_set:
        if not line.strip():
            # sentence has ended
            result = viterbi(sentence_buffer, observations, labels, A, B)
            for index, word in enumerate(sentence_buffer):
                file.write(f"{word} {result[index]}")

            file.write("\n")
        else:
            sentence_buffer.append(line)
    file.close()

if __name__ == "__main__":
    for locale in ["EN", "FR", "CN", "SG"]:

        DATA = open(f"./../data/{locale}/train")
        training_set = part2.prepare_data(DATA)
        _results, observations, label_counts, emission_counts = part2.estimate_emissions(
            training_set)

        TEST_DATA = open(f"./../data/{locale}/dev.in")
        testing_set = part2.prepare_data(TEST_DATA)
        # with the test data, we are able to smooth out the emissions.
        emissions = part2.smooth_emissions(
            testing_set, observations, label_counts, emission_counts)

        label_counts_with_start_stop, A = estimate_transitions(training_set)

        B = part2.get_B(label_counts_with_start_stop, emissions)

        predict_viterbi(locale, A, B)