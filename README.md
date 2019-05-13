# COS 424 Final Project: Applying Online and Reinforcement Learning Techniques to Improve Cache Performance
Aninda Manocha

Note: This project uses both Python 2.7 and Python 3.6. The alias "python" runs Python 2.7, so if your "python" alias runs Python 3.5, then you need to ensure you are using Python 2.7 where the command below is running with the "python" alias.

## Preprocessing The Data

The original data format for this project is a memory trace of a program that contains (at the least) the memory access addresses and the program counters. Additional information such as the instruction type could be added to the data. Since there are not many features in the feature space, the data needs to be preprocessed in order to create a more informative feature space. There are three modes of doing so:

- Address Clustering* (`cluster`): k-means clustering with `$NUMBER OF CLUSTERS` is performed on the address space and addresses are labeled by the cluster they are assigned to
- Address Space Division (`divide`): the address space is divided evenly into `$NUMBER OF CLUSTERS` clusters and addresses are labeled by the cluster they fall in
- Address Frequency Threshold* (`threshold`): only addresses that appear in the memory trace `$PREPROCESS_THRESHOLD` times maintain their labels while all other addresses receive a new label to denote them as addresses that don't appear frequently enough

* These modes are not typically practical for online learning because the entire memory trace needs to be analyzed beforehand. One possibility would therefore be to analyze and preprocess the memory trace, and then later perform online learning with the addresses in the memory trace one at a time.

After the addresses have been relabeled, one-hot encoding is performed on the newly generated labels to create a larger feature space. If `$PERFORM_DIMENSION_REDUCTION` is 1, then PCA is performed on the new feature space as a means of applying dimension reduction and filtering out features that might not be as informative for cache access prediction. If `$PERFORM_DIMENSION_REDUCTION` is 0, then no dimension reduction is performed.

    python3 preprocess.py [--file $PATH_TO_DATA_FILE] [--mode $PREPROCESS_MODE] [--num_clusters $NUMBER OF CLUSTERS/DIVISIONS] [--threshold $PREPROCESS_THRESHOLD] [--dim_red $PERFORM_DIMENSION_REDUCTION]

## Applying Reinforcement Learning

## Applying Other Online Learning Algorithms with Classifiers
