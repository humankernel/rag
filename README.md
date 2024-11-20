# Advanced RAG Pipeline 

<a href="https://colab.research.google.com/github/humankernel/rag/blob/main/notebooks/colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="vertical-align: middle;"/></a> <a href="https://rcci.uci.cu/?journal=rcci" target="_parent"><img src="https://badgen.net/badge/paper/Open/red" alt="Open Paper" style="vertical-align: middle;"/></a>

![alt text](assets/objects.png)

## Overview 

This project is a prototype developed for a research paper focused on democratizing AI tools for managing PDF documents in resource-limited contexts. It serves as an advanced Retrieval-Augmented Generation (RAG) pipeline example, showcasing how AI can facilitate document management.

## Features

- **Gradio UI**: An interactive user interface for easy interaction with the RAG pipeline.
- **Testing Framework**: Utilizes the RAGET library from Griskard to ensure robust testing of functionalities.
- **Advanced RAG Pipeline**: Demonstrates state-of-the-art techniques for document management and retrieval.


## Installation

To set up the project locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/humankernel/rag.git
cd rag
```

2. (Optional) Set up a virtual environment to manage dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install the required dependencies:

```sh
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

4. Open the notebook (`notebooks/rag.ipynb`)

5. (Optional) Setup pypi cuban repos

```ini
# edit
# linux: ~/.config/pip/pip.conf
# windows: ~\AppData\Roaming\pip\pip.ini

[global]
timeout = 120
index = http://nexus.prod.uci.cu/repository/pypi-all/pypi
index-url = http://nexus.prod.uci.cu/repository/pypi-all/simple
[install]
trusted-host = nexus.prod.uci.cu
```

## Usage

Since this is currently a prototype all of the main code is in a single jupyter notebook 

`rag.ipynb`: notebook with rag pipeline and evaluations
`colab.ipynb`: colab ready notebook
`chunking.ipynb`: exploration of different chunking strategies
`indexing.ipynb`: exploration of different indexing strategies
`retrieval.ipynb`: exploration of different retrieval strategies
`diagrams.ipynb`: code for creating the diagrams of the paper

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/YourFeature).
3. Make your changes and commit them (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/YourFeature).
5. Open a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details
