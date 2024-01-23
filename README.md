# Text Improvement Engine

## Overview
Text Improvement Engine is a tool designed to analyze and suggest improvements to text documents. It leverages a machine learning model to compare phrases in a given text against a set of standard phrases, providing suggestions to align the text more closely with these standards.

## Requirements
- Python 3.10
- [Poetry](https://python-poetry.org/)

## Installation
```
poetry install
```
Might also need to select the proper Python version with:
```
poetry env use <path_to_python_executable>
```

## Components
The tool consists of two main components:
1. **CLI Application (`app_cli.py`)**: A command-line interface for analyzing text files.
2. **GUI Application (`app_gui.py`)**: A graphical user interface, built with PyQt5, for a more interactive text analysis experience.

## Usage

### CLI Application

To use the CLI application, run `app_cli.py` with the path to the text file as an argument:

```
poetry run python text_improvement_engine/app_cli.py <path_to_text_file>
```

This will output the original phrases from the text, suggested standard phrases, and their similarity scores.

### GUI Application

Run `app_gui.py` to start the GUI application:

```
poetry run python text_improvement_engine/app_gui.py
```

In the GUI, you can input the text directly, click 'Analyse' to process the text, and view the suggestions in a table format.

It's possible to play around with `MIN_NGRAM_LENGTH` and `MAX_NGRAM_LENGTH` values in `engine.py`, but there will be a performance cost.

## Known Issues

- The tool may exhibit slower performance on very large text files due to the computational demands of generating and comparing embeddings. Also, running it for the first time will be slower due to the need to download/install required packages. 
- There are large limitations in contextually and grammatically fitting the suggestions into the overall text structure. The engine doesn't analyze the complete semantic tree of the text, which means suggested phrases might not always conjugate or integrate seamlessly into the original sentence structure. This limitation stems from the engine's focus on phrase-level analysis rather than a comprehensive linguistic understanding of the entire text.
