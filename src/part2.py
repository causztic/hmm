UNKNOWN_TOKEN = "#UNK#"

# generate a list to use from the data.


def prepare_data(file):
    from itertools import groupby
    lines = [line for line in file]
    chunks = (list(g) for k, g in groupby(lines, key=lambda x: x != '\n') if k)
    return [[word.rstrip('\n') for word in chunk] for chunk in chunks]

# estimates the emission parameters from the given sequence.


def estimate_emissions(sequence):
    label_counts = {}  # count of every unique tag
    emission_counts = {}  # count of tag -> observation
    results = {}  # MLE results
    words = set()  # track the words in the training set

    # flatten the list
    sequence = (item for sublist in sequence for item in sublist)

    for item in sequence:
        pair = item.rsplit(" ", 1)

        words.add(pair[0])

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

    return results, words, label_counts, emission_counts

# estimates the emission parameters from the given sequence from the testing set
# and a given set of words from the training set.


def smooth_emissions(sequence, words, label_counts, emission_counts):
    results = {}  # MLE results
    k = 1  # set k to 1 according to question

    # flatten the list
    sequence = (item for sublist in sequence for item in sublist)

    for item in sequence:
        if item not in words:
            # new word, add to count of unknowns
            k += 1

    print(f"There are {k} new words in the testing set.")
    # If the word token x appears in the training set. i.e. all existing emission_counts.
    for key, value in emission_counts.items():
        values = key.rsplit(" ", 1)
        probability = float(value) / (label_counts[values[-1]] + k)

        if values[0] in results:
            results[values[0]][values[1]] = probability
        else:
            results[values[0]] = {values[1]: probability}

    # If word token x is the special token #UNK#. i.e. for every label, we just add in a new condition #UNK#|y.
    # This would be 0 if there are no #UNK#.
    results[UNKNOWN_TOKEN] = {}

    for key, value in label_counts.items():
        results[UNKNOWN_TOKEN][key] = float(k) / (value + k)

    return results


def predict_tags(locale, results):
    """Get most probable label -> observation, and write to file."""
    tags = {}
    training_set = [line.rstrip("\n")
                    for line in open(f"./../data/{locale}/dev.in")]

    for key, value in results.items():
        highest = -1
        for tag, prob in value.items():
            if prob > highest:
                highest = prob
                tags[key] = tag

    file = open(f"./../data/{locale}/dev.p2.out", "w")
    for line in training_set:
        if not line.strip():
            file.write("\n")
        else:
            if line in tags:
                tag_value = tags[line]
            else:
                tag_value = tags[UNKNOWN_TOKEN]

            file.write(f"{line} {tag_value}\n")
    file.close()


if __name__ == "__main__":
    for locale in ["EN", "FR", "CN", "SG"]:

        DATA = open(f"./../data/{locale}/train")
        training_set = prepare_data(DATA)
        _results, words, label_counts, emission_counts = estimate_emissions(
            training_set)

        TEST_DATA = open(f"./../data/{locale}/dev.in")
        testing_set = prepare_data(TEST_DATA)
        # with the test data, we are able to smooth out the emissions.
        results = smooth_emissions(
            testing_set, words, label_counts, emission_counts)

        # we perform argmax on each word to get the most probable tag, and perform prediction.
        predict_tags(locale, results)
