# nameExtractor


## Installation
Clone or fork this repository, open a terminal window and in the directory where you downloaded nameextractor, type the following commands   
for Linux
```shell
python -m venv ne
source ne/bin/activate
pip install -e .
```  

for windows:
```shell
python -m venv ne
ne/Scripts/activate 
pip install -e .
```

Download the spacy model:
```
python -m spacy download en_core_web_sm
```

To deactivate the virtual environment:
```
deactivate
```

# Usage
```python
from extractors.name_extractor import NameExtractor

name_extractor = NameExtractor()
name_extractor.extract(text=text)
```
