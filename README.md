# Learning Flags - Server
Client source code: https://github.com/ruurdbijlsma/LearningFlagsClient

## Requirements
* Python3
* Pip for Python3

## Installation

```
git clone https://github.com/RuurdBijlsma/LearningFlagsServer
cd LearningFlagsServer
pip install -r requirements.txt
```

## Running the server
```
cd src
python main.py
```

## Generate requirements.txt
Make sure you have pip-tools `python -m pip install pip-tools`

`pip-compile --output-file=requirements.txt requirements.in`