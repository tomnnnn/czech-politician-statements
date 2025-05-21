# Factual Checking and Reliability of Sources from Open Media
**Author**: Hai Phong Nguyen

## Czech Statements Dataset
This repository contains scripts to build the Czech politician statements dataset and an automatic fact-checker used to evaluate it.

The following is provided:
- scripts to build the dataset
- scripts to run fact-checking evaluation on the dataset
- scripts to optimize the fact-checkers prompts
- experiment evaluation results for the fact-checking pipeline

---
#### Installing Dependencies
I recommend a conda or mamba environment. Otherwise, the installation steps are standard for a Python application.
```
mamba create -n factcheck python=3.12
mamba activate  factcheck
pip install -r requirements.txt
```

---
#### Displaying Existing Evaluation Results

To display the experiment results used in the thesis, use:

```
mlflow ui --backend-store-uri sqlite:///thesis-evals.db
```

Additionally, preliminary experiments can be viewed using the tool located in `utils/benchmark_viewer`:

```
streamlit run utils/benchmark_viewer/Benchmark.py
```

---
#### Building the Dataset

To build the dataset on your own, run:
```
PYTHONPATH=src python -m src.dataset_builder
```

Configuration file can be found in `src/dataset_builder/config/config.yaml`:

```
  # general options
  OutputDir: datasets/datasetT
  FetchConcurrency: 3
  FetchDelay: 1

  # Demagog scraping options
  UseExistingStatements: false
  FromYear: null
  ToYear: null
  FirstNPages: null

  # Evidence retrieval options
  UseExistingEvidenceLinks: false
  ScrapeArticles: True
  EvidenceNum: 2
  SearchDelay: 1
  SearchesPerDelay: 3
  EvidenceRetriever: demagog
  EvidenceAPIKey: $SEARCH_API_KEY
  SegmentArticles: False
```

In the default configuration, the script scrapes all the statements across all the years from Demagog.cz and scrapes the links provided in the fact-check explanations to form the evidence repository.

_Besides from getting evidence links from Demagog itself, the dataset builder is capable to search for evidence using Google or Bing search API (note that the relevancy of these evidence is rather low, as it only uses the statements as the search query)_

Article segments for hybrid retrieval can be created by enabling `SegmentArticles` in config, or by using:

```
PYTHONPATH=src python -m src.segmenter <dataset_file_path>
```

##### Auto Curation

Due to the imperfect scraping, many statements might not have enough evidence to be reliably verifiable. Automatic curation aims to lessen this factor. To do it, run:

```
PYTHONPATH=src python -m src.fact_checker.auto_curate
```

---

#### Creating Hybrid Retrieval Index
To create the hybrid retrieval indexes, run:

```
./build_index.sh -d <dataset_path> -o <output_folder>
```

---

#### Running the Experiments

##### Setup

To run the experiments, the evidence retriever server must be running. You can start the server by:

```
uvicorn src.retriever_api.app:app --port 4242
```

The server can be configured using environment variables:
```
export RETRIEVER_DATASET_PATH= ...
export RETRIEVER_INDEX_PATH= ...
```

##### Experiments

The experiments are configured to use this setting by default:
- Retriever API url: `localhost:4242`
- LLM OpenAI-compatibile API: `localhost:8000/v1`
- LiteLLM model string: `hosted_vllm/Qwen/Qwen2.5-32B-Instruct-AWQ`

This can be configured through environment variables:
```
export LLM_BASE_API= ...
export LLM_MODEL= ...
export RETRIEVER_BASE_API= ...

# optionally
export OPENAI_API_KEY= ...
```

Experiments with the fact-checking pipeline can be run using the `run_experiments.sh` script, or manually by running the experiments scripts in `src/evaluate`. Their results are saved to `results` folder.

```
PYTHONPATH=src python -m src.evaluate.baseline
```

#### Prompt Optimization

To optimize the prompts for the LLM calls using the dspy's MIPROv2 method, run this:

```
PYTHONPATH=src python -m src.optimize
```


---

This repository is a part of the bachelor thesis _Factual checking and reliability of sources from open media_ on FIT BUT 
