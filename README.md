# AI Summarization
Repo showcasing AI summarization tool.

## Summary
This repo showcases a simple, yet effective tool for document summarization. It can work with plain-text and PDF documents in any language supported by underlying LLM (Mistral by default).

## Setup

### Installing Dependencies

Install following dependencies (on macOS):

- Python 3 installation (e.g. [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Homebrew package](https://formulae.brew.sh/formula/python@3.10))
- Python packages: run `pip3 install -r requirements.txt`
- Download `mistral-7b-openorca.Q5_K_M.gguf` model from Hugging Face [TheBloke/Mistral-7B-OpenOrca-GGUF](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/tree/main) repo into local `models` directory.

Note you can experiment with anternatives models, just update the `MODEL_FILE` and `MODEL_CONTEXT_WINDOW` variables in `web-ui.py` and/or `Notebook.ipynb`.

## Running

### Web UI

In order to run Web UI just run `python3 ./web-ui.py` in the repo folder. This should open Web UI interface in the browser.

### Jupyter Notebook

The tool can be used as Jupyter Labs/Notebook as well, you open the  `Notebook.ipynb` in [Jupyter Labs](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html#conda).

## Details

### Workflow

Depending on the document size, this tool works in following modes:
1. In the simple case, if the whole document can fit into model's context window then summarizartion is based on adding relevant summarization prompt.
2. In case of large documents, document processed using "map-reduce" pattern:
  1. The document is first split into smaller chunks using `RecursiveCharacterTextSplitter`` which tries to respect paragraph and sentence boundarions.
  2. Each chunk is summarized separately (map step).
  3. Chunk summarization are summarized again to give final summary (reduce step).

### Local processing
All processing is done locally on the user's machine.
- Quantified Mistral model (`mistral-7b-openorca.Q5_K_M.gguf`) has around 5,1 GB.

### Performance

Relatively small to medium documents (couple of pages) should fit into single context window, which results in processing time of around 40s on Apple MBP with M1 chip.

## Troubleshooting

None know issue.
