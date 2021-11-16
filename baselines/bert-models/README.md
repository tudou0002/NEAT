Use the following command to run all BERT-based baseline models, RoBERTA and HT_bert for person name extraction

```
python transformer_bert.py --dataset \path\to\tsv\file\with\ads --model \path\to\fine-tuned\BERT\model --inp_column_name "name of text column in your file" --res_column_name "column name for extracted results"
```

To use transformer-bert model from HuggingFace fine-tuned for NER on CoNLL2003 dataset, omit the `--model` parameter.

To use other BERT-based fine-tuned models, first follow the instructions from [`fine-tuning/`](fine-tuning/README.md/) and then rerun this command.


To run the experiment for general NER,

```
python transformer_bert_general.py --dataset \path\to\tsv\file\with\ads --model \path\to\fine-tuned\BERT\model --res_column_name "column name for extracted results"
```