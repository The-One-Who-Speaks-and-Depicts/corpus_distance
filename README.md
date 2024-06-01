# Python package for measuring distance between the lects represented by small raw corpora
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11395683.svg)](https://doi.org/10.5281/zenodo.11395683)

![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)

# What is it?

corpus_distance is a Python package that allows to measure distance between the lects that are presented only by small (down to extremely small, <1000 tokens) raw (without any kind of morhological tagging, lemmatisation, or dependency parsing) corpora, and classify them. It joins frequency-based metrics and string similarity measurements into a hybrid distance scorer.

corpus_distance operates with 3-shingles, a sequences of 3 symbols, by which words are split. This helps to spot more intricate patterns and correspondences within raw data, as well as to enhance the dataset size.

## NB!

The classification is going to be only (and extremely) preliminary, as it is by default language-agnostic and does not use preliminary expert judgements or linguistic information. Basically, the more effort is put into the actual data, the more reliable are final results. 

In addition, the results may not be used as a proof of language relationship (external classification), only as a supporting evidence for a tree topology (internal classification), as it is with any kind of phylogenetic methods in historical comparative studies.

One more important notion is that one should be very careful with using this package for a distantly related lects. As it is with any kind of language-agnostic methods, it loses precision with the increase of distance between analysed groups (Holman et al., 2008). 

# How to install

## From TestPyPI (development version; requires manual installation of dependencies; may contain bugs)

```
pip install biopython
pip install Levenshtein
pip install gensim
pip install pyjarowinkler

python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps --force-reinstall corpus_distance
```

## From PyPI (release version)

```
pip install corpus_distance
```

## From Docker

1. Clone repository.
```
git clone https://github.com/The-One-Who-Speaks-and-Depicts/corpus_distance.git
```
2. Cd to the repository root directory.
3. (optionally) Add your data and configuration file (see How to use/Preparation below), put both into the repository directory.
4. Create Docker image.
```
sudo docker build --tag IMAGE_NAME .
```
IMAGE_NAME is a name of the docker image of your choice
5. Run Docker image.
```
sudo docker run -i -t IMAGE_NAME /bin/bash
```

# How to use

## Preparation

1. Create a virtual environment with `python3 -m venv ENV`, where `ENV` is a name of your virtual environment.
2. Install package, following the instructions above.
3. Create a folder for your data.
4. Put files with your texts into this folder. Texts should be tokenised manually, and joined into a single string afterwards. The name of the files with texts should be of the following format: TEXT.LECT.EXTENSION, where: 
   - EXTENSION is preferrably .txt, as package works with files with raw text data;
   - LECT is a name of the lect (idiolect, dialect, sociolect, regiolect, standard, etc.; any given variety of the language, such as English, or Polish, or Khislavichi, or Napoleon's French) that is the object of the classification
   - TEXT is a unique identifier of the text within a given lect (for instance, NapoleonSpeech1, or John_Gospel)
5. Set up a configuration .json file (the example is in the repository). The parameters are:
   -  `store_path`: a path to the folder for results storage
   -  `content_path`: a path to the data folder
   -  `split`: a share of tokens from your files that would be taken into consideration (useful for exploring size effects)
   -  `lda_params`: a set of parameters for a Latent Dirichlet Association model from `gensim` package
   -  `topic_modelling`: model may delete topic words, if this flag has value `true`, or not, if value is `false`. This heuristic helps to exclude the words that define the text, on the contrary to the ones that define the language
   -  `fasttext_params`: a set of parameters for a FastText model that provides the classifier with the symbol embeddings
   -  `soerensen`: normalisation of frequency-based metrics by the Soerensen-Dice coefficient
   -  `hybridisation`: flag for use (or not use) of string similarity measure for non-coinciding 3-shingles
   -  `hybridisation_as_array`: regulates the way of hybridisation: either frequency-based metrics and string similarity measures values are taken as a single array, for which the mean score is counted, or they are taken separately, and their means are multiplied by each other. `soerensen` normalisation applies only when this parameter has `false` value.
   -  `metrics`: a particular string similarity measure. May be user defined, defaults are `corpus_distance.distance_measurement.string_similarity.levenshtein_wrapper` (simple edit distance), `corpus_distance.distance_measurement.string_similarity.weighted_jaro_winkler_wrapper`(edit distance, weighted by Jaro-Winkler distance), `corpus_distance.distance_measurement.string_similarity.vector_measure_wrapper` (counting differences by euclidean distance between vectors of symbols), and `corpus_distance.distance_measurement.string_similarity.jaro_vector_wrapper` (counting differences by euclidean distance between vectors of symbols, weighted by Jaro distance, in order to count for order)
   -  `alphabet_normalisation`: a normalisation of vector-based metrics by difference of alphabet entropy between given lects
   -  `data_name`: name of the dataset for visualisation (for example. South Slavic)
   -  `outgroup`:  name of the lect that is the farthest from the others
   -  `metrics`: name of the metrics combination, by default containing all the given parameters
   -  `classification_method`: classification method for building tree, `upgma` or `nj`: either Unweighted Pair Group Method with Arithmetic Mean, or Neighbourhood-Joining
   -  `store_path`: the same as `store path` on the top.
6. The example of the `config.json`:
```
    {
        "store_path": "default",
        "metrics_name": "default_metrics_name",
        "data": {
            "content_path": "default",
            "split": 1,
            "lda_params": {
                "num_topics": 10,
                "alpha": "auto",
                "epochs": 300,
                "passes": 500
            },
            "topic_modelling": false,
            "fasttext_params": {
                "vector_size": 128,
                "window": 15,
                "min_count": 3,
                "workers": 4,
                "epochs": 300,
                "seed": 42,
                "sg": 1
            }
        },
        "hybridisation_parameters": {
            "soerensen": true,
            "hybridisation": true,
            "hybridisation_as_array": true,
            "metrics": "corpus_distance.distance_measurement.string_similarity.jaro_vector_wrapper",
            "alphabet_normalisation": true
        },
        "clusterisation_parameters": {
            "data_name": "Modern Standard Slavic",
            "outgroup": "Slovak",
            "metrics": "default_metrics_name",
            "classification_method": "upgma",
            "store_path": "default"
        }
    }
```
7. In case you are using Docker, do not forget to put your data and configuration file into the repository directory before creating Docker image. 

## Running the code

There are two ways of running the code: with prepared Jupyter Notebook, or independently.

### Ready-made Jupyter Notebook

In the folder `example`, there is a tutorial notebook that outlines the inner workings of the package.

### Using your own file (or from Docker console)

After data and configuration are ready, open Python interpreter:

```
python
```

Run the following commands:

```
from corpus_distance.pipeline import perform_clusterisation
perform_clusterisation(PATH_TO_CONFIG)
```

`PATH_TO_CONFIG` here is a path to `config.json`. This parameter may be empty, then the model performs clusterisation with default dataset (Modern Standard Slavic Gospels of John: Croatian, Slovak, Slovenian) and default parameters.
