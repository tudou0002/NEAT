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

Extract the model for disambiguation:
```bash
tar -xzvf ht_bert_v3.tar.gz
```

To deactivate the virtual environment:
```
deactivate
```

# Usage
There are two modules: name extractor and dictionary expansion. 
### Name extractor
```python
from extractors.name_extractor import NameExtractor

# initialize a NameExtractor instance 
# default primary extractors as dictionary and crf extractor
# default backoff extractors is the rule extractor
name_extractor = NameExtractor(weights_dict='extractors/src/weights.json')
text = "I'm Andriana,  Waiting to play with you "
# there's a default text preprocessing step in the extract method
results = name_extractor.extract(text=text)
# return a list of Entity identified by the extractor

# Inspect on the entity
for ent in results:
    print('Entity text:', ent.text)
    print('Entity confidence score:', ent.confidence)
# Entity text: Andriana
# Entity confidence score: 0.235
```
