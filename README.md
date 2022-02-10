# NEAT
This is the repository for **NEAT**(Name Extraction Against Trafficking). NEAT is designed for extracting person names from escort ads. It effectively combines classic  rule-based and dictionary extractors with a contextualized language model to capture ambiguous names and adapts to adversarial changes in the text by expanding its dictionary. **NEAT** shows 23% improvement on average in the F1 classification score for name extraction compared to previous state-of-the-art in two domain-specific datasets.

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
 
### Name extractor
```python
from extractors.name_extractor import NameExtractor

# initialize a NameExtractor instance 
name_extractor = NameExtractor(weights_dict='extractors/src/weights.json')
text = "My name is Andriana."
# there's a default text preprocessing step in the extract method
results = name_extractor.extract(text=text)
# return a list of Entity identified by the extractor

# Inspect on the entity
for ent in results:
    print('Entity text:', ent.text)
    print('Entity confidence score:', ent.confidence)
# Entity text: Andriana
# Entity confidence score: 0.7175
```
