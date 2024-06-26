# Diffusion Models Encode the Intrinsic Dimension of Data Manifolds
![ICML 2024](https://img.shields.io/badge/ICML-2024-blue.svg)

This repo is the official pytorch implementation of the paper [Diffusion Models Encode the Intrinsic Dimension of Data Manifolds](https://arxiv.org/abs/2212.12611).

by Jan Stanczuk*, Georgios Batzolis*, Teo Deveney, and Carola-Bibiane Schönlieb

You can find the paper on [arXiv](https://arxiv.org/abs/2212.12611) and more details on the project's [website](https://gbatzolis.github.io/ID-diff/).


--------------------

In this work, we provide a mathematical proof that diffusion models encode data manifolds by approximating their normal bundles. Based on this observation we propose a novel method for extracting the intrinsic dimension of the data manifold from a trained diffusion model. Our insights are based on the fact that a diffusion model approximates the score function i.e. the gradient of the log density of a noise-corrupted version of the target distribution for varying levels of corruption. We prove that as the level of corruption decreases, the score function points towards the manifold, as this direction becomes the direction of maximal likelihood increase. Therefore, at low noise levels, the diffusion model provides us with an approximation of the manifold's normal bundle, allowing for an estimation of the manifold's intrinsic dimension. To the best of our knowledge, our method is the first diffusion-based estimator of intrinsic dimension that comes with theoretical guarantees, and it outperforms well-established estimators in controlled experiments on both Euclidean and image data.

## How to run the code

### Dependencies

Run the following to install a subset of necessary python packages for our code
```sh
pip install -r req.txt
```

### Usage

Train and evaluate our models through `main.py`.

```sh
main.py:
  --config: Training configuration.
    (default: 'None')
  --mode: <train|manifold_dimension>: Running mode: train or manifold_dimension
```

* `config` is the path to the config file. The config files used in our official experiments are in the folder configs/dimension_estimation/paper

* `mode` is either "train" or "manifold_dimension". When set to "train", it starts the training of a new model, or resumes the training if config.model.checkpoint_path is not None. When set to "model_dimension", it estimates the ID of the dataset using the checkpoint provided in config.model.checkpoint_path

## References

If you find the code useful for your research, please consider citing
```bib
@article{stanczuk2022your,
  title={Your diffusion model secretly knows the dimension of the data manifold},
  author={Stanczuk, Jan and Batzolis, Georgios and Deveney, Teo and Sch{\"o}nlieb, Carola-Bibiane},
  journal={arXiv preprint arXiv:2212.12611},
  year={2022}
}
```
