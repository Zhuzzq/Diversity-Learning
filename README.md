## Diversity Learning: A Space-time Scheme for Ensemble Learning

Diversity framework provides a space-time weighting scheme for ensemble learning.

The 2×1 scheme is implemented.

![img](https://github.com/Zhuzzq/Diversity-Learning/blob/master/framework.png)

### Run


- The E-combiner and several models are implemented in `diversity_models.py`
- Run `ori_train.py` with arguments to train the single models on CIFAR-10.
- Run `transfer_diversity_only_E.py` with arguments to train the E-combiner or weighting ensemble module.

### Reference


* *Diversity Learning: Introducing the Space-time Scheme to Ensemble Learning*
