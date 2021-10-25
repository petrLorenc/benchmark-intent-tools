If you use the data and publish please cite our preprint paper:

``` 
@article{lorenc2021benchmark,
  title={Benchmark of public intent recognition services},
  author={Lorenc, Petr and Marek, Petr and Pichl, Jan and Konr{\'a}d, Jakub and {\v{S}}ediv{\`y}, Jan},
  journal={Language Resources and Evaluation},
  pages={1--19},
  year={2021},
  publisher={Springer}
}
```

# README
This project is a collection of code and datasets for evaluating intent classification models (local and remote). The results can be helpful when creating chatbots or other conversational interfaces.

## Structure

### code/nlu_local

Code for performing local testing.

#### code/nlu_local/approaches

Different approaches for classification

### code/nlu_analysers and code/nlu_converters

Updated code from [NLU-Evaluation-Corpora](https://github.com/sebischair/NLU-Evaluation-Corpora)


### code/nlu_local/embeddings

Template for getting sentence embeddings - due to privacy issue cannot be provided.

### data

Three dataset used in [NLU-Evaluation-Corpora](https://github.com/sebischair/NLU-Evaluation-Corpora) - used to perform comparison.

#### data/corpus

Own data for evaluation

#### data/results

Result in JSON format

#### data/results/visualisation

Confusion matrix visualisation - using [Facets](https://pair-code.github.io/facets/)

## Contact Information
If you have any questions, please contact:

**Petr Lorenc** (Czech Technical University) petr.lorenc@cvut.cz
