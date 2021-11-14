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
ne\Scripts\activate.bat
pip install -e .
```

Download the spacy model:
```
python -m spacy download en_core_web_sm
# the transformer model that can reach the state-of-the-art performance in multiple NER tasks
python -m spacy download en_core_web_trf
```

To deactivate the virtual environment:
```
deactivate
```

# Usage
```python
from extractors.name_extractor import NameExtractor

# initialize a NameExtractor instance 
# default primary extractors as dictionary and crf extractor
# default backoff extractors is the rule extractor
name_extractor = NameExtractor()
text = "I'm Andriana,  Waiting to play with you  My Body Is Your Playground  Your time with me will be a full non rushed session fitted to your needs and desires. No Drama . No Hassle .No black GENTS ! I wan't to enjoy our time together  Come Shower With Me!! I am 100% Independent and am here for YOU!!! Let Me Create Your Dream Fantasy Find out why I'll have you coming back for more Always available to answeer you baby"
# there's a default text preprocessing step in the extract method
name_extractor.extract(text=text)
# return a list of strings with unqiue names identified by the extractor
```
