# show-your-work
Code for the "Show Your Work" (paper)[https://arxiv.org/abs/1909.03004], EMNLP 2019.

## Citation
If you use this repository for your research, please cite:

```
@inproceedings{showyourwork,
 author = {Jesse Dodge and Suchin Gururangan and Dallas Card and Roy Schwartz and Noah A. Smith},
 title = {Show Your Work: Improved Reporting of Experimental Results},
 year = {2019},
 booktitle = {Proceedings of EMNLP},
}
```

## Installation

First, clone this repository.

Then, clone the ``allentune`` repository and install it on your system:

```
git clone https://github.com/allenai/allentune
cd allentune/
pip install --editable .
```


## Run example

```
cd show-your-work/
allentune search --experiment-name cnn_search \
                 --search_space ./search_spaces/cnn_sst5.json \
                 --base-config ./training_config/cnn_classifier.jsonnet \
                 --cpus-per-trial 1 \
                 --num-gpus 0 \
                 --gpus-per-trial 0  \
                 --num-samples 1
                 --include-package show-your-work
```

If you have GPUs, set the `--num-gpus` flag and `--gpus-per-trial` flag appropriately.


**Note:** To run DGEM search on Scitail, you must use the implementation in https://github.com/allenai/scitail, which notably depends on a previous
version of allennlp. Clone that repository, copy the `search_spaces/dgem.json` and `training_config/dgem.jsonnet` files over, and run

```
allentune search --experiment-name dgem_search \
                 --search_space ./search_spaces/dgem.json \
                 --base-config ./training_config/dgem.jsonnet \
                 --cpus-per-trial 1 \
                 --num-gpus 0 \
                 --gpus-per-trial 0  \
                 --num-samples 1
                 --include-package scitail
```
