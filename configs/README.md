HDPGMM training configurations
=====

In this folder, we generate and keep the configurations that are used for the inference (training) of the HDPGMM models.

The configurations are generated according to the experimental design for the study; in this particular experiment, we unfortunately are not able to setup the either full or fractional design due to the resource limitation. What we do instead is the sequentially dependent design, where we somewhat greedly employ the prior knowledge from literature to fix some of the hpyer parameter and find the optimal parameter that we introduce. Then, we go back to those parameters that are fixed by the prior knowledge to find out whether we can find different best ones or not.


## Parameters

The set of hyper-parameters we want to tune are as following:

1. regularization (noise): $\alpha$ (isn't it already occupied?)
2. learning rate parameters: $\kappa, \tau0$
3. batch size

As descrived earlier, we start to tune the noise addition factor $\{0, 1e-4, 1e-3, 1e-2, 1e-1\}$ and then moving on to the 2nd and 3rd ones while fixing the optimal noise addition parameter.

Although we do expect that the noise addition would interact with the learning rate and batch size parameters in general[^1], we leave the more comprehensive experiment for the future works.


## Other notes...

To study the model fit variance occurred by the random initialization, we try to replicate the runs as many as possible. We generate 5 replications each per case, and run up to the point we can compute.



[^1]: as more regularization often slows down the inference to be converged and thus optimal learning rate / batch size will be varying with respect to the regularization setup.
