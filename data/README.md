# Dataset setup

This project uses the UCI Drug Review Dataset (Drugs.com), originally published
with predefined training and test splits. Download it from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com)
and place the files here:

```text
data/drugsComTrain_raw.csv
data/drugsComTest_raw.csv
```

Expected columns are `uniqueID`, `drugName`, `condition`, `review`, `rating`,
`date`, and `usefulCount`. The CSV files are ignored by Git; review the UCI
dataset terms before redistributing them.
