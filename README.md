# HMM 🤔

## Part 2 - Emission-only predictions
To predict with emissions-only, the training set has all the lines concatentated together and read through, word by word. A set is used to keep track of unique observations, while maintaining the counts of each observation and label in a map.

After going through all the words in the training set, we perform MLE to obtain a set of each observation and their probabilities of being generated by every label.

The data is smoothed by the testing set, adding in a special #UNK# token for words that have not appeared in the training set as the last item in the observation list. We keep track of the count of #UNK# as k, and used that as explained in the question.

We now take the labels that has the highest possibility of generating each particular word by applying argmax across all words, after which we naively apply the results to the testing set.

To run, do:
`cd src && python3 part2.py`

|                       | EN    | FR    | SG     | CN
| --------------------- | --    | --    | --     | --
| #Entity in gold data  | 802   | 238   |4092    |1081
| #Entity in prediction | 1129  | 836   |10784   |4545
| #Correct Entity       | 620   | 191   |2443    |594
| Entity  precision     | 0.5492| 0.2285|0.2265  |0.1307
| Entity  recall        | 0.7731| 0.8025|0.5970  |0.5495
| Entity  F             | 0.6422| 0.3557|0.3284  |0.2112
| #Correct Entity Type  | 458   | 105   |1322    |364
| Entity Type  precision| 0.4057| 0.1256|0.1226  |0.0801
| Entity Type  recall   | 0.5711| 0.4412|0.3231  |0.3367
| Entity Type  F        | 0.4844| 0.1955|0.1777  |0.1294

## Part 3 - First order transition probabilities and Viterbi
To perform vitberi, we have to calculate the transition probabilities.
We have a matrix of states (labels), of size K, where K is the number of states.
The 0th-axis is Si, while 1st-axis is Sj, such that a value in the matrix (i,j) would be the probability of Si transitioning to Sj. K does not include START and STOP, instead we have two extra arrays of size K to keep the transitions from START -> Si, and Si -> STOP.

We have the dataset split into sentences and iterated through in a sliding window size of 2. We add to the frequencies of Si -> Sj, and to the counts of each state appearing.

After constructing these frequencies, we perform MLE to obtain the probabilities of each transition and store in the matrix. We used the emission probabilities created in Part 2 as well. We reference the pseudocode on Wikipedia to perform the Viterbi algorithm, but added in a Si -> STOP evaluation and used log likelihood to compute the probabilities instead of the default multiplication to reduce numerical underflow.

To run, do:
`cd src && python3 part3.py`

|                       | EN    | FR    | SG     | CN
| --------------------- | --    | --    | --     | --
| #Entity in gold data  | 802   | 238   |4092    |1081
| #Entity in prediction | 734   | 202   |3558    |1556
| #Correct Entity       | 509   | 110   |1646    |432
| Entity  precision     | 0.6935| 0.5419|0.4626  |0.2776
| Entity  recall        | 0.6347| 0.4622|0.4022  |0.3996
| Entity  F             | 0.6628| 0.4989|0.4303  |0.3276
| #Correct Entity Type  | 460   | 61    |1057    |302
| Entity Type  precision| 0.6267| 0.3005|0.2971  |0.1941
| Entity Type  recall   | 0.5736| 0.2563|0.2583  |0.2794
| Entity Type  F        | 0.5990| 0.2766|0.2763  |0.2290

## Part 4 - Second order transition probabilities and Viterbi
For the second order transition, we have a 3d matrix, where the 0th-axis is Si, 1st-axis is Sj, and 2nd-axis is Sk, and Si, Sj -> Sk. The procedure is roughly the same as Part 3, except with a sliding window of 3 to capture the state transitions.

We referred to this paper here http://www.arnaud.martin.free.fr/publi/PARK_14a.pdf for fine-tuning. They mentioned that it is quite unlikely for test data to have the same second order transitions as the training data, it can be normalized using a method of guessing - deleted interpolation. This ensures that the second order transitions are weighted more, but we have some parts of first order and zero order transitions to normalize the data as well in the event that Sk|Si,Sj does not exist.

The Viterbi algorithm is modified to be a 2d matrix every step instead of an array in Part 3 to accomodate an extra state in the propagation.

To run, do:
`cd src && python3 part4.py`

|                       | EN    | FR
| --------------------- | --    | --    
| #Entity in gold data  | 802   | 238   
| #Entity in prediction | 874   | 745   
| #Correct Entity       | 477   | 88   
| Entity  precision     | 0.5458| 0.1181
| Entity  recall        | 0.5948| 0.3697
| Entity  F             | 0.5692| 0.1790
| #Correct Entity Type  | 370   | 57
| Entity Type  precision| 0.4233| 0.0765
| Entity Type  recall   | 0.4613| 0.2395
| Entity Type  F        | 0.4415| 0.1160

## Part 5

For part 5, we will be using structured perceptrons. Instead of using MLE for an averaged transition and emission weights and probabilities, we have each line evaluated as-is and update the weights as the algorithm iterates through. For the first iteration of the algorithm, the emission probabilities and transition probabilities are used as the starting weight of the parameters. 
As the algorithm iterates through, the weight is added (+1) if the predicted observation for a particular tag matches that of the training set’s observations for a given tag. Should the predicted observations be different from the training set, the weight is penalized (-1). The larger weight gives rise to a higher likelihood of the predicted observation being the same as the training set and thus a greater likelihood of the prediction to be accurate. 
The algorithm is iterated 4 times.


To run, open Jupyter Notebook and open the file at `src/ML Project part 5.ipynb`.

|                       | EN    | FR
| --------------------- | --    | --    
| #Entity in gold data  |    |    
| #Entity in prediction |    |    
| #Correct Entity       |    |    
| Entity  precision     |    | 
| Entity  recall        |    | 
| Entity  F             |    | 
| #Correct Entity Type  |    | 
| Entity Type  precision|    | 
| Entity Type  recall   |    | 
| Entity Type  F        |    | 
