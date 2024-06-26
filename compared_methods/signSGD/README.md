# signSGD: compressed optimisation for non-convex problems

<img src="https://jeremybernste.in/publications/signum/norms.png" width="120" align="right"></img>

Here I house mxnet code for the original signSGD paper (ICML-18). I've put the code here to facilitate reproducing the results in the paper, and this code isn't intended for development purposes. In particular, this implementation does not gain any speedups from compression. Some links:
- [arxiv version of the paper](https://arxiv.org/abs/1802.04434).
- more information about the paper on [my personal website](https://jeremybernste.in/publications/).
- my coauthors: [Yu-Xiang Wang](https://www.cs.cmu.edu/~yuxiangw/), [Kamyar Azizzadenesheli](https://sites.google.com/uci.edu/kamyar), [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/).

**[Update Jan 2021]** As noted in [this issue](https://github.com/jxbz/signSGD/issues/1), this codebase used an implementation of the sign function that maps `sign(0) --> 0`. A test in [this notebook](https://github.com/jxbz/signSGD/blob/master/signSGD_zeros.ipynb) suggests there may be little difference to an implementation that maps `sign(0) --> ±1` at random. In the [codebase](https://github.com/jiaweizzhao/signSGD-with-Majority-Vote) for the [ICLR 2019 paper](https://arxiv.org/abs/1810.05291), we used an implementation that maps `sign(0) --> +1` deterministically.

***

General instructions:
- Signum is implemented as an official optimiser in mxnet, so to use Signum in this codebase, we pass in the string 'signum' as a command line argument.
- if you do not use our suggested hyperparameters, be careful to tune them yourself. 
- Signum hyperparameters are typically similar to Adam hyperparameters, not SGD!

***

There are four folders:

1. cifar/ -- code to train resnet-20 on Cifar-10.
2. gradient_expts/ -- code to compute gradient statistics as in Figure 1 and 2. Includes [Welford algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance?oldformat=true#Online_algorithm).
3. imagenet/ -- code to train resnet-50 on Imagenet. Implementation inspired by that of [Wei Wu](https://github.com/tornadomeet/ResNet).
4. toy_problem/ -- simple example where signSGD is more robust than SGD.

More info to be found within each folder.

***

Any questions / comments? Don't hesitate to get in touch: <a href="mailto:bernstein@caltech.edu">bernstein@caltech.edu</a>.
