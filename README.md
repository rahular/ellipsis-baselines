## Coreference and Ellipsis as QA
Code to reproduce the experiments in [A Simple Transfer Learning Baseline for Ellipsis Resolution](https://arxiv.org/abs/1908.11141)

### Conversion and Utilities
The repository contains conversion scripts for converting different datasets into the SQuAD 1.1 format.

* `vpe2squad.py`: Convert VP ellipsis dataset into SQuAD format
* `conll2squad.py`: Convert coreference data from C0NLL-2012 to SQuAD format
    - First convert `.conll` files to `.jsonlines` using [this](https://github.com/kentonl/e2e-coref/blob/master/minimize.py)
    - Set `ONTONOTES_DIR` (ontonotes folder path) and `set2fmt` (filename to convert to SQuAD format)
    - Run script
* `sluice2squad.py`: Convert sluice ellipsis dataset into SQuAD format
* `wikicoref2conll.py`: Convert WikiCoref dataset into CoNLL-2012 format
* `squad2conll.py`: Convert the prediction files produced by `bert/run_squad.py` into CONLL format for evaluation

#### Miscellaneous
* `annotate_qwords.py`: Adds `<ref>` and `</ref>` tags to interrogation words in SQuAD files
* `evaluate-v1.1.py`: Standard SQuAD v1.1 evaluation script (for evaluating ellipsis) 

For coreference resolution, use the standard [CoNLL-2012 script](http://conll.cemantix.org/2012/software.html) after converting the predictions into the CoNLL-2012 format using `squad2conll.py`.

### Training Details
Each model folder contains pre-processing, configuration, training and evaluation scripts for Sluice Ellipsis. To run on other datasets, just replace the data paths appropriately.

#### DrQA
* Code based on Facebook's [DrQA](https://github.com/facebookresearch/DrQA)
* Scripts for [preprocessing](https://github.com/coastalcph/universal-qa/blob/release/DrQA/run/prepare-data.sh), [training](https://github.com/coastalcph/universal-qa/blob/release/DrQA/run/train.sh) and [prediction](https://github.com/coastalcph/universal-qa/blob/release/DrQA/run/predict.sh)

#### QAnet
* Code based on [AllenNLP](https://github.com/allenai/allennlp)
* AllenNLP [configuration file](https://github.com/coastalcph/universal-qa/blob/release/qanet/qanet_sluice.json)
* Scripts for [training](https://github.com/coastalcph/universal-qa/blob/release/qanet/scripts/train.sh) and [prediction](https://github.com/coastalcph/universal-qa/blob/release/qanet/scripts/predict.sh)

#### BERT
* Uses Huggingface's [Transformers](https://github.com/huggingface/transformers)
* Scripts for [training](https://github.com/coastalcph/universal-qa/blob/release/bert/scripts/train.sh) and [evaluation](https://github.com/coastalcph/universal-qa/blob/release/bert/scripts/predict.sh)

