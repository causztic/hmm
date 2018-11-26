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
| #Entity in prediction | 712   | 202   |3349    |4384
| #Correct Entity       | 490   | 109   |1706    |434
| Entity  precision     | 0.6882| 0.5396|0.5094  |0.0990
| Entity  recall        | 0.6110| 0.4580|0.4169  |0.4015
| Entity  F             | 0.6473| 0.4955|0.4585  |0.1588
| #Correct Entity Type  | 443   | 60    |1108    |296
| Entity Type  precision| 0.6222| 0.2970|0.3308  |0.0675
| Entity Type  recall   | 0.5524| 0.2521|0.2708  |0.2738
| Entity Type  F        | 0.5852| 0.2727|0.2978  |0.1083

## Part 4 - Second order transition probabilities and Viterbi
`cd src && python3 part4.py`

## Part 5
