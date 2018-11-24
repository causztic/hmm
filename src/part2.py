import numpy as np
from itertools import groupby

UNKNOWN_TOKEN = "#UNK#"

def prepare_data(file):
    """
    Prepare the file, and returns a list of lists of "{observation} {label}"

    file : the name of the file to read
    """

    lines = [line for line in file]
    chunks = (list(g) for k, g in groupby(lines, key=lambda x: x != '\n') if k)
    return [[observation.rstrip('\n') for observation in chunk] for chunk in chunks]

def get_observation_set(sequence, add_unknown_token=False):
    """
    Reads the sequence and returns a set() of observations.

    sequence : a list of lists of either "{observation} {label}" or "{observation}"
    """

    observation_set = set()
    if add_unknown_token:
        observation_set.add(UNKNOWN_TOKEN)

    # flatten the lists
    sequence = (item for sublist in training_sequence for item in sublist)

    for item in sequence:
        observation = item.rsplit(" ", 1)[0]
        observation_set.add(item)

    return observation_set

def get_label_set(training_sequence):
    """
    Reads the sequence and returns a set() of labels.

    training_sequence : a list of lists of "{observation} {label}"
    """

    label_set = set()
    # flatten the lists
    training_sequence = (item for sublist in training_sequence for item in sublist)

    for item in training_sequence:
        label = item.rsplit(" ", 1)[1]
        label_set.add(label)

    return label_set

def estimate_emissions(sequence):
    """
    Estimates the emission parameters from the given training sequence, without smoothing.

    sequence : a list of lists of "{observation} {label}"
    """

    label_counts = {}  # count of every unique label
    emission_counts = {}  # count of label -> observation
    results = {}  # MLE results
    observations = set()  # track the observations in the training set

    # flatten the list
    sequence = (item for sublist in sequence for item in sublist)

    for item in sequence:
        observation, label = item.rsplit(" ", 1)
        observations.add(observation)
        if item in emission_counts:
            emission_counts[item] += 1
        else:
            emission_counts[item] = 1
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    # here we estimate the emission parameters using MLE after obtaining the counts.
    # count(label -> observation) / count(label)
    for key, emission_count in emission_counts.items():
        observation, label = key.rsplit(" ", 1)
        results[f"{label} -> {observation}"] = emission_count / float(label_counts[label])

    return results, list(observations), label_counts, emission_counts

def smooth_emissions(sequence, observations, label_counts, emission_counts):
    """
    Estimates the emission parameters from the sequence (all the sentences) from the testing set,
    and a given set of observations from the training set.

    sequence       : a list of lists of "{observation} {label}" from the testing set
    observations   : observations from the training set
    label_counts   : { label: count } from the training set
    emission_counts: { "{observation} {label}": count } from the training set
    """
    labels = list(label_counts)
    B = np.zeros((len(label_counts), len(observations) + 1)) # + 1 to accomodate for UNKNOWN_TOKEN.
    k = 1  # set k to 1 according to question

    # flatten the list
    sequence = (item for sublist in sequence for item in sublist)

    for item in sequence:
        if item not in observations:
            # new observation, add to count of unknowns
            k += 1

    # If the observation appears in the training set i.e. it appeared in emission_counts.
    for key, emission_count in emission_counts.items():
        observation, label = key.rsplit(" ", 1)
        probability = float(emission_count) / (label_counts[label] + k)

        B[labels.index(label), observations.index(observation)] = probability

    # If observation is #UNK#. i.e. for every label, we just add in a new condition #UNK#|y.
    # This would be 0 if there are no #UNK#.

    for label, label_count in label_counts.items():
        B[labels.index(label), -1] = float(k) / (label_count + k)

    return B

def predict_labels(locale, B, observations, label_counts):
    """
    Get most probable label -> observation, and write to file.

    locale       : locale of the dataset. should be either SG, EN, CN, or FR
    B            : K by K matrix of emission probabilities.
    observations : a list of observations in the training data
    label_counts : { label -> count }
    """
    labels = list(label_counts)
    training_set = [line.rstrip("\n")
                    for line in open(f"./../data/{locale}/dev.in")]


    file = open(f"./../data/{locale}/dev.p2.out", "w")
    for line in training_set:
        if not line.strip():
            file.write("\n")
        else:
            if line in observations:
                # if the observation is in our observations, we take the most probable label.
                label_value = labels[np.argmax(B[:,observations.index(line)])]
            else:
                # take the unknown's value.
                label_value = labels[np.argmax(B[:,-1])]

            file.write(f"{line} {label_value}\n")
    file.close()


if __name__ == "__main__":
    for locale in ["EN", "FR", "CN", "SG"]:

        DATA = open(f"./../data/{locale}/train")
        training_set = prepare_data(DATA)
        _results, observations, label_counts, emission_counts = estimate_emissions(
            training_set)

        TEST_DATA = open(f"./../data/{locale}/dev.in")
        testing_set = prepare_data(TEST_DATA)
        # with the test data, we are able to smooth out the emissions.
        B = smooth_emissions(
            testing_set, observations, label_counts, emission_counts)

        # we perform argmax on each observation to get the most probable label for each observation.
        predict_labels(locale, B, observations, label_counts)