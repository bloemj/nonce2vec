# nonce2vec
[![GitHub release][release-image]][release-url]
[![PyPI release][pypi-image]][pypi-url]
[![Build][travis-image]][travis-url]
[![MIT License][license-image]][license-url]

Welcome to Nonce2Vec!

This branch of the repository accompanies the Bloem et al. (2020) manuscript [*Distributional Semantics for New Latin*].

**If you are looking for the Kabbach et al (2019) version of Nonce2Vec with English test sets, check out the [main Nonce2Vec repository](https://github.com/minimalparts/nonce2vec).**

If you use this code, please cite:
```tex
@inproceedings{kabbach-etal-2019-towards,
    title = "Towards Incremental Learning of Word Embeddings Using Context Informativeness",
    author = "Kabbach, Alexandre  and
      Gulordava, Kristina  and
      Herbelot, Aur{\'e}lie",
    booktitle = "Proceedings of the 57th Conference of the Association for Computational Linguistics: Student Research Workshop",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-2022",
    pages = "162--168"
}
```

**Abstract**

**

## A note on the code
The code in this repository is largely the same as Nonce2Vec v2, but with additional code for evaluating consistency as demonstrated in the Bloem et al. (2019) RANLP paper [*Evaluating the Consistency of Word Embeddings from Small Data*](https://www.aclweb.org/anthology/R19-1016/). For more information on the software, visit the [main Nonce2Vec repository](https://github.com/minimalparts/nonce2vec).

## Install
After downloading and extracting, you can install Nonce2Vec via:
```bash
python3 setup.py install
```

## Pre-requisites
To run Nonce2Vec on Latin data, you need a gensim Word2Vec model trained on Latin data.
<!---(a skipgram model and a cbow model to compute informativeness-metrics). You can download the skipgram model from:
```bash
wget http://129.194.21.122/~kabbach/gensim.w2v.skipgram.model.7z
```
and the cbow model from:
```sh
wget http://129.194.21.122/~kabbach/gensim.w2v.cbow.model.7z
```--->
You can generate them yourself using the instructions below.

### Generating the Vicipaedia background model
<!---You can download our English Wikipedia dump of January 2019 here:
```bash
wget http://129.194.21.122/~kabbach/enwiki.20190120.7z
```--->
A tokenized-one-sentence-per-line dump of Wikipedia, for Latin or any other language, can be generated using [WiToKit](https://github.com/akb89/witokit).

Once you have a Vicipaedia txt dump, you can generate a gensim Word2Vec skipgram model via:
```bash
n2v train \
  --data /absolute/path/to/vicipaedia/tokenized/text/dump \
  --outputdir /absolute/path/to/dir/where/to/store/w2v/model \
  --alpha 0.1 \
  --neg 5 \
  --window 15 \
  --sample 1e-4 \
  --epochs 5 \
  --min-count 50 \
  --size 400 \
  --num-threads number_of_cpu_threads_to_use \
  --train-mode skipgram
```
The specified parameters are the best-performing ones from our Definitional evaluation in the paper. The min-count and size parameters can be varied to obtain the different experimental conditions in Table 2.

<!---and a gensim Word2Vec cbow model via:
```bash
n2vconsistency train \
  --data /absolute/path/to/wikipedia/tokenized/text/dump \
  --outputdir /absolute/path/to/dir/where/to/store/w2v/model \
  --alpha 0.025 \
  --neg 5 \
  --window 5 \
  --sample 1e-3 \
  --epochs 5 \
  --min-count 50 \
  --size 400 \
  --num-threads number_of_cpu_threads_to_use \
  --train-mode cbow
```-->

### Creating the evaluation sets

The evaluation sets are included in this release in [resources/](resources/).
The Definitional evaluation and tuning set can be regenerated (with a different random sampling of terms) using the sampling script [preprocessing/sample-wiki-n2vevalset.py](preprocessing/sample-wiki-n2vevalset.py).

The New Latin Mathematical Methods evaluation set (shown in Table 4) was manually compiled. The provenance of the included text passages can be found in [data/](data/).

## Running the code
Running Nonce2Vec on the definitional or New Latin datasets is done via the `n2v test` command. You can pass in the `--reload` parameter to run in `one-shot` mode, without it the code runs in incremental model by default. You can further pass in the `--shuffle` parameter to shuffle the test set before running n2v.

You will find below a list of commands corresponding to the experiments reported in the manuscript. For example, to test the sum baseline model, run, for the definitional dataset:
```bash
n2v test \
  --on latdeftest \
  --model /absolute/path/to/gensim/w2v/skipgram/model \
  --sum-only \
  --sum-filter random \
  --sample 500 \
  --window 5 \
  --sample-decay 1.9 \
  --window-decay 5 \
  --replication
```
And for the 18th century mathematical methods New Latin dataset:
```bash
n2v test \
  --on mm18thtest \
  --model /absolute/path/to/gensim/w2v/skipgram/model \
  --sum-only \
  --sum-filter random \
  --sample 500 \
  --window 5 \
  --sample-decay 1.9 \
  --window-decay 5 \
  --replication
```

To test N2V on the Latin definitional dataset using the tuned parameters, do:
```bash
n2v test \
  --on latdeftest \
  --model /absolute/path/to/gensim/w2v/skipgram/model \
  --sum-filter random \
  --sample 500 \
  --alpha 0.5 \
  --neg 3 \
  --window 5 \
  --epochs 1 \
  --lambda 70 \
  --sample-decay 1.9 \
  --window-decay 5 \
  --replication
```
This runs the original (Herbelot & Baroni, 2017) N2V code without background freezing and without CWI-based informativity, in incremental setup.

To test N2V on the New Latin dataset using the tuned parameters, do:
```bash
n2v test \
  --on mm18thtest \
  --model /absolute/path/to/gensim/w2v/skipgram/model \
  --sum-filter random \
  --sample 500 \
  --alpha 0.5 \
  --neg 3 \
  --window 5 \
  --epochs 1 \
  --lambda 70 \
  --sample-decay 1.9 \
  --window-decay 5 \
  --replication
```
This performs a consistency evaluation using the code of the Bloem et al. (2019) RANLP paper [*Evaluating the Consistency of Word Embeddings from Small Data*](https://www.aclweb.org/anthology/R19-1016/).

## Background models for the New Latin consistency evaluation

As shown in Table 2, we compare the use of background models trained over different Latin corpora. The procedure to create the Vicipaedia model is described above. As for the other models:

### Bamman corpus

The Bamman corpus is [described here](https://www.cs.cmu.edu/~dbamman/latin.html). We do not use the pre-trained word embeddings, as they are not in a format that can be trained further. We downloaded [the plain-text version](https://docs.google.com/uc?id=0B5pGKi0iCsnbZEdHZ3N6d216am8&export=download) and tokenized it with [Ucto](https://languagemachines.github.io/ucto/) as described in the paper.

Next, in the same way as for the Wikipedia texts, we generate a gensim Word2Vec skipgram model via:
```bash
n2v train \
  --data /absolute/path/to/bamman/tokenized/text/dump \
  --outputdir /absolute/path/to/dir/where/to/store/w2v/model \
  --alpha 0.025 \
  --neg 5 \
  --window 5 \
  --sample 1e-3 \
  --epochs 5 \
  --min-count 50 \
  --size 400 \
  --num-threads number_of_cpu_threads_to_use \
  --train-mode skipgram
```
The specified parameters are the Nonce2Vec defaults and not tuned for small data, as this dataset is much larger even than the English Wikipedia dump used by Herbelot & Baroni (2017). The min-count and size parameters can be varied to obtain the different experimental conditions shown in Table 2.

### Latin Text Library corpus
We used the [version of the Latin Text Library provided by the Classical Languages Toolkit in plain-text format](https://github.com/cltk/latin_text_latin_library). We tokenized it with Polyglot using the the script [preprocessing/tokeniser_polyglot_latin.py](preprocessing/tokeniser_polyglot_latin.py) and [removed punctuation](preprocessing/alnum_latin.py).

Next, in the same way as for the Wikipedia texts, we generate a gensim Word2Vec skipgram model via:
```bash
n2v train \
  --data /absolute/path/to/lattextlib/tokenized/text/dump \
  --outputdir /absolute/path/to/dir/where/to/store/w2v/model \
  --alpha 0.1 \
  --neg 5 \
  --window 15 \
  --sample 1e-4 \
  --epochs 5 \
  --min-count 50 \
  --size 400 \
  --num-threads number_of_cpu_threads_to_use \
  --train-mode skipgram
```
The specified parameters are the best-performing ones from our Vicipaedia Definitional evaluation in the paper, as this dataset is of a comparable size to Vicipaedia. The min-count and size parameters can be varied to obtain the different experimental conditions in Table 2.

### Treebank corpora
This corpus is a concatenation of the Index Thomisticus, Perseus, and PROIEL treebanks, all collected in the context of the Universal Dependencies project. We used the CoNLLU format files from the UD Latin [Index Thomisticus](https://github.com/UniversalDependencies/UD_Latin-ITTB), [Perseus](https://github.com/UniversalDependencies/UD_Latin-Perseus) and [PROIEL](https://github.com/UniversalDependencies/UD_Latin-PROIEL) repositores, and extracted from these files all the words in the second column (unlemmatized tokens).

Next, in the same way as for the Wikipedia texts, we generate a gensim Word2Vec skipgram model via:
```bash
n2v train \
  --data /absolute/path/to/treebanks/token/text/dump \
  --outputdir /absolute/path/to/dir/where/to/store/w2v/model \
  --alpha 0.1 \
  --neg 5 \
  --window 15 \
  --sample 1e-4 \
  --epochs 5 \
  --min-count 50 \
  --size 400 \
  --num-threads number_of_cpu_threads_to_use \
  --train-mode skipgram
```
The specified parameters are the best-performing ones from our Vicipaedia Definitional evaluation in the paper, as this dataset is small. The min-count and size parameters can be varied to obtain the different experimental conditions in Table 2.

[release-image]:https://img.shields.io/github/release/minimalparts/nonce2vec.svg?style=flat-square
[release-url]:https://github.com/minimalparts/nonce2vec/releases/latest
[pypi-image]:https://img.shields.io/pypi/v/nonce2vec.svg?style=flat-square
[pypi-url]:https://pypi.org/project/nonce2vec/
[travis-image]:https://img.shields.io/travis/akb89/nonce2vec.svg?style=flat-square
[travis-url]:https://travis-ci.org/akb89/nonce2vec
[license-image]:http://img.shields.io/badge/license-MIT-000000.svg?style=flat-square
[license-url]:LICENSE.txt
