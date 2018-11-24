from itertools import groupby

UNKNOWN_TOKEN = "#UNK#"

def prepare_data(file):
    """Prepare the file. Generates a list of lists of sentences"""
    lines = [line for line in file]
    chunks = (list(g) for k, g in groupby(lines, key=lambda x: x != '\n') if k)
    return [[observation.rstrip('\n') for observation in chunk] for chunk in chunks]

def estimate_emissions(sequence):
    """Estimates the emission paramters from the given sequence."""

    label_counts = {}  # count of every unique label
    emission_counts = {}  # count of label -> observation
    results = {}  # MLE results
    observations = set()  # track the observations in the training set

    # flatten the list
    sequence = (item for sublist in sequence for item in sublist)

    for item in sequence:
        pair = item.rsplit(" ", 1)

        observations.add(pair[0])

        if item in emission_counts:
            emission_counts[item] += 1
        else:
            emission_counts[item] = 1
        if pair[1] in label_counts:
            label_counts[pair[1]] += 1
        else:
            label_counts[pair[1]] = 1

    for key, value in emission_counts.items():
        values = key.rsplit(" ", 1)
        results[f"{values[0]}|{values[1]}"] = value / \
            float(label_counts[values[1]])

    return results, observations, label_counts, emission_counts

def smooth_emissions(sequence, observations, label_counts, emission_counts):
    """
    Estimates the emission parameters from the sequence (all the sentences) from the testing set,
    and a given set of observations from the training set.
    """

    results = {}  # MLE results
    k = 1  # set k to 1 according to question

    # flatten the list
    sequence = (item for sublist in sequence for item in sublist)

    for item in sequence:
        if item not in observations:
            # new observation, add to count of unknowns
            k += 1

    print(f"There are {k} new observations in the testing set.")
    # If the observation token x appears in the training set. i.e. all existing emission_counts.
    for key, value in emission_counts.items():
        values = key.rsplit(" ", 1)
        probability = float(value) / (label_counts[values[-1]] + k)

        if values[0] in results:
            results[values[0]][values[1]] = probability
        else:
            results[values[0]] = {values[1]: probability}

    # If observation token x is the special token #UNK#. i.e. for every label, we just add in a new condition #UNK#|y.
    # This would be 0 if there are no #UNK#.
    results[UNKNOWN_TOKEN] = {}

    for key, value in label_counts.items():
        results[UNKNOWN_TOKEN][key] = float(k) / (value + k)

    return results

def get_B(label_counts, emissions):
    # from the results, we generate a K x N matrix.
    B = np.zeros((len(label_counts), len(emissions)))

    # emissions structure is something like
    # { observation: { label_1: n, label_2: n2 }}

    for j, (j_key, j_value) in enumerate(emissions.items()):
        for i, i_key in enumerate(label_counts.keys()):
            if i_key in j_value:
                # if the label emits the observation at j
                B[i, j] = j_value[i_key]
    return B

def predict_labels(locale, results):
    """Get most probable label -> observation, and write to file."""
    labels = {}
    training_set = [line.rstrip("\n")
                    for line in open(f"./../data/{locale}/dev.in")]

    for key, value in results.items():
        highest = -1
        for label, prob in value.items():
            if prob > highest:
                highest = prob
                labels[key] = label

    file = open(f"./../data/{locale}/dev.p2.out", "w")
    for line in training_set:
        if not line.strip():
            file.write("\n")
        else:
            if line in labels:
                label_value = labels[line]
            else:
                label_value = labels[UNKNOWN_TOKEN]

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
        results = smooth_emissions(
            testing_set, observations, label_counts, emission_counts)

        # we perform argmax on each observation to get the most probable label for each observation.
        predict_labels(locale, results)
