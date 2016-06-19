# EvoML
Can evolutionary sampling improve bagged ensembles?

##Introduction
Bagging and various variants of the it have been widely popular and studied extensively in the last two decade. There has been
notable work in understanding the theoretical underpinning of bootstrap aggregating and as to what makes it such
a powerful method. 

In traditional bagging, each training example is sampled with replacement and with probability N1 . Adaptive Resampling and Combining techniques which modify the probability of each training example being sampled based on heuristics have also been developed and widely used.

#Motivation
Random sampling, Error based resampling algorithms which try to set the train-set error to zero, designed bagged ensembles with minimal intersection (Papakonstantinou et al., 2014), diversity and uncorrelated errors, importance sampling etc. are some of the areas being studied to improve bagged ensembles. Either there are multiple answers to the question, or the answer changes with each dataset.

##How?
Instead of figuring out precisely as to what sampling and combination of training sets make a bagged ensemble better, we try to fix the definition of better, and allow the bootstrapped training sets to evolve themselves in order to align with the definition.

We generate multiple sampled candidate training sets for the final ensemble and let them compete, mutate and mate their way to the optimal sampling
and combination

We use Evolutionary sampling in two domains `Subsampling` which is sampling rows of data and `Subspacing` which is sampling features.


##Playground
We've developed basic Evolutionary Sampling based prediction models for both `subspacing` and `subsampling` seperately. The API is built upon sklearn's fit and predict API.
Currently we are experimenting with 3 different `fitness` functions:

- FEMPO (Fitness Each Model Private OOB)
- FEGT (Fitness Ensemble Global Test)
- FEPT (Fitness Each Model Private Test)

You can read about them in our paper, *Can evolutionary sampling improve bagged ensembles?*

##Example
Add link to a demo notebook


##Contribute
In the spirit of `reproduciblity` we've kept the research open and would be thrilled to have contributors and collaboraters to the research. Please get in touch any ideas or submit an issue if you find a bug.

##License
GNU GENERAL PUBLIC LICENSE

##Authors
Harsh Nisar and Bhanu Pratap