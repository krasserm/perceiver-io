## Update

After having written the initial version of the [article](https://krasserm.github.io/2023/01/23/scaling-perceiver-ar/)
I found that Andrej Karpathy reported a problem with FLOPs computation (Appendix F) in the Chinchilla paper.
**Other than stated in the paper, token embedding matrices do not contribute to training FLOPs calculation** (details
[here](https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb) and [here](https://news.ycombinator.com/item?id=34342427)).

After implementing the [required changes](https://github.com/krasserm/perceiver-io/commit/59d550051d9991bab3c9cc97cdc853249df0af6b#diff-0a2a46a7a80ffcaca5250dfa995765ccccea96a0e15b394e5b66af8c9ebfeffb)
on branch [wip-scaling](https://github.com/krasserm/perceiver-io/tree/wip-scaling) it turned out that the [conclusion](https://krasserm.github.io/2023/01/23/scaling-perceiver-ar/#conclusion)
of the article still remains valid although some numbers changed, of course. These changes made it necessary to [train
a new compute-optimal model](https://github.com/krasserm/perceiver-io/blob/wip-scaling/examples/scaling/clm/train.md#model-training)
for experiment 1 and the results are shown in a [partially updated version of the article](https://github.com/krasserm/perceiver-io/blob/wip-scaling/examples/scaling/clm/article.ipynb).
The final loss of the compute-optimal model is still lower than that of the reference model. Training new models for
experiments 2a and 2b was not necessary as explained further below.

I initially also struggled reproducing the exact numbers in some tables of the paper but thought this was related to
the contribution of softmax and value reduction to FLOPs calculation. I omitted these, as done in \[2\], but included
the embedding matrices as described in \[1\]. This resulted in $C / C_{approx}$ ratios that were very close to those
reported in the paper.

The main difference between my initial implementation of FLOPs calculation and the updated implementation is the
omission of the final dense layer i.e. the layer that shares its weights with the token embedding layer. This
leads to a smaller compute estimate compared to the initial version, especially for smaller models with a larger
vocabulary. This is the reason why it had a stronger impact on experiment 1 (vocabulary size = 32,000) than on
experiments 2a and 2b (vocabulary size = 262). For larger models, the relative contribution of embedding matrices
is small anyway.

The calculation of model size was already done correctly in the initial version of the article. Here, the final dense
layer contributes to the calculation (details [here](https://github.com/krasserm/perceiver-io/blob/59d550051d9991bab3c9cc97cdc853249df0af6b/examples/scaling/clm/scaling/flops.py#L152-L157)
and [here](https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb)). There are only minor differences
because of positional embeddings and some bias terms.

## References

\[1\] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks,
J. Welbl, A. Clark, et al. Training compute-optimal large language models.
[arXiv preprint arXiv:2203.15556](https://arxiv.org/abs/2203.15556), 2022.

\[2\] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu,
and D. Amodei. Scaling laws for neural language models.
[arXiv preprint arXiv:2001.08361](https://arxiv.org/abs/2001.08361), 2020.
