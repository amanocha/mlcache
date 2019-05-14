# COS 424 Final Project: Applying Online and Reinforcement Learning Techniques to Improve Cache Performance
Aninda Manocha

Note: This project uses both Python 2.7 and Python 3.6. The alias "python" runs Python 2.7, so if your "python" alias runs Python 3.5, then you need to ensure you are using Python 2.7 where the command below is running with the "python" alias.

## Gathering the Data

The datasets for this project were memory traces of programs. The data used in this project consisted of memory traces of two programs:
1. Graph Projections - a graph algorithm that takes a bipartite graph as input and relates nodes in one partition based on neighbors they share in the other partition
    - This algorithm was run on data gathered on crime in Moreno Valley: http://konect.uni-koblenz.de/downloads/tsv/moreno_crime.tar.bz2.
    - For more information on this algorithm, visit https://en.wikipedia.org/wiki/Bipartite_network_projection.
2. Random Pointer Chase - a benchmark program that involves many random memory accesses and pointer chasing

To obtain this data, visit https://drive.google.com/drive/folders/1bLXzAv1Vel5uU0ZLOWg82WfILrnvkUAb?usp=sharing.
## Preprocessing The Data

The original data format for this project is a memory trace of a program that contains (at the least) the memory access addresses and the program counters. Additional information such as the instruction type could be added to the data. Since there are not many features in the feature space, the data needs to be preprocessed in order to create a more informative feature space. There are three modes of doing so:

- Address Clustering* (`cluster`): k-means clustering with `$NUMBER OF CLUSTERS` is performed on the address space and addresses are labeled by the cluster they are assigned to
- Address Space Division (`divide`): the address space is divided evenly into `$NUMBER OF CLUSTERS` clusters and addresses are labeled by the cluster they fall in
- Address Frequency Threshold* (`threshold`): only addresses that appear in the memory trace `$PREPROCESS_THRESHOLD` times maintain their labels while all other addresses that don't appear frequently enough don't receive any label

* These modes are not typically practical for online learning because the entire memory trace needs to be analyzed beforehand. One possibility would therefore be to analyze and preprocess the memory trace, and then later perform online learning with the addresses in the memory trace one at a time.

After the addresses have been relabeled, one-hot encoding is performed on the newly generated labels to create a larger feature space. If `$PERFORM_DIMENSION_REDUCTION` is 1, then PCA is performed on the new feature space as a means of applying dimension reduction and filtering out features that might not be as informative for cache access prediction. If `$PERFORM_DIMENSION_REDUCTION` is 0, then no dimension reduction is performed.

    python3 preprocess.py --file $PATH_TO_DATA_FILE --mode $PREPROCESS_MODE [--num_clusters $NUMBER OF CLUSTERS/DIVISIONS] [--threshold $PREPROCESS_THRESHOLD] [--dim_red $PERFORM_DIMENSION_REDUCTION]

## Applying Reinforcement Learning

Once the data has been preprocessed, it is ready to fed into the models. To perform cache access prediction with no prefetching, run the following:

    python bandits_base.py --file $PATH_TO_DATA_FILE --model_num $MODEL_NUMBER
    
where `$MODEL_NUMBER` must be one of the following:
- `0`: Adaptive Greedy Threshold
- `1`: Adaptive Greedy Percentile
- `2`: Adaptive Active Greedy
- `3`: Bootstrapped Thompson Sampling
- `4`: Bootstrapped Upper-Confidence Bounds
- `5`: Softmax

To perform prediction-based prefetching, run the following:

    python bandits.py --file $PATH_TO_DATA_FILE --model_num $MODEL_NUMBER --distance $PREFETCH_DISTANCE
    
where `$MODEL_NUMBER` is the same as above and `$PREFETCH_DISTANCE` is the number of cycles ahead of time a prediction is to be made. The maximum distance is 250 cycles, which is the latency of fetching data from main memory.

There are bash scripts to automate the experimentation process with all reinforcement learning models:
- `bandits_bash.sh` - perform cache access prediction with no prefetching for all six variants of contextual bandits (useful for evaluating the models themselves with metrics such as accuracy, false positive rate, false negative rate, etc.)
- `bandits.sh` - perform prediction-based prefetching with a prefetch distance of 250 cycles for all six variants of contextual bandits (useful for evaluating cache performance)
- `bandits.sh` - perform prediction-based prefetching with a prefetch distance of 250 cycles for all six variants of contextual bandits using a larger (32 KB vs. 2 KB) cache (useful for improving the original hit rate)

* Note that running the contextual bandits algorithms requires Python 2.7.

## Applying Online Classification

To perform cache access prediction with no prefetching, run the following:

    python3 classify_base.py --file $PATH_TO_DATA_FILE --model_num $MODEL_NUMBER

where `$MODEL_NUMBER` must be one of the following:
- `0`: Decision Tree
- `1`: Random Forest
- `2`: K-Nearest Neighbors
- `3`: Naïve Bayes

To perform prediction-based prefetching, run the following:

    python3 classify.py --file $PATH_TO_DATA_FILE --model_num $MODEL_NUMBER --distance $PREFETCH_DISTANCE
    
where `$MODEL_NUMBER` is the same as above and `$PREFETCH_DISTANCE` is the number of cycles ahead of time a prediction is to be made. The maximum distance is again 250 cycles.

There are also bash scripts to automate the experimentation process with all classifiers:
- `classify_base.sh` - perform cache access prediction with no prefetching for all four classifiers (useful for evaluating the models themselves with metrics such as accuracy, false positive rate, false negative rate, etc.)
- `classify.sh` - perform prediction-based prefetching with a prefetch distance of 250 cycles for all four classifiers (useful for evaluating cache performance)
