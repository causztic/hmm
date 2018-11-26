# HMM 🤔

## Part 2 - Emission-only predictions
`cd src && python3 part2.py`

|  | EN | FR | SG | CN |
|- | -- | -- | -- | -- |
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
`cd src && python3 part3.py`

|  | EN | FR | SG | CN |
|- | -- | -- | -- | -- |
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
`cd src && python3 part4.py`

## Part 5
