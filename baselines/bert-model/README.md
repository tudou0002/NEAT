## Fine-tuning BERT models for the baselines

In order to run the baseline-models, do the following:

`cd fine_tuning`


1. `transformer-bert` - this baseline does not require any fine-tuning and uses a pre-trained bert-base-cased NER pipeline model. Refer to the instructions of running `tr_bert.py`. (```python tr_bert.py --help```).

2. `fine-tuned-bert` - this baseline fine-tunes a bert-base-cased model on a mix of CoNLL (500 samples) + Listcrawler dataset (200 samples) for the task of NER. To do this, run ```python run_ner.py --data_dir data/ --model_type bert --labels data/labels.txt --model_name_or_path bert-base-cased --output_dir fine_tuned_bert/ --max_seq_length 128 --num_train_epochs 5 --per_gpu_train_batch_size 8 --save_steps 100 --logging_steps 100 --seed 42 --do_train --overwrite_output_dir ```

3. `ht-bert` - this baseline fine-tunes a masked language model on HT data and further fine-tunes it for the task of NER using the same mixed dataset from above. The fine-tuned masked language model on HT data should be obtained separately (due to large file size). To fine-tune for NER, run ```python run_ner.py --data_dir data/ --model_type bert --labels data/labels.txt --model_name_or_path "path to the language model trained on HT data obtained separately" --output_dir ht_bert/ --max_seq_length 128 --num_train_epochs 5 --per_gpu_train_batch_size 8 --save_steps 100 --logging_steps 100 --seed 42 --do_train --overwrite_output_dir ```

4. `whole-mask-bert` - this baseline fine-tunes a whole-word-masked bert model for the task of NER on the same mixed dataset from above. The intuition is that a bert model trained by masking whole words would make more sense for name extraction. To fine-tune for NER, run ```python run_ner.py --data_dir data/ --model_type bert --labels data/labels.txt --model_name_or_path bert-large-cased-whole-word-masking --output_dir whole_mask_bert/ --max_seq_length 128 --num_train_epochs 5 --per_gpu_train_batch_size 8 --save_steps 100 --logging_steps 100 --seed 42 --do_train --overwrite_output_dir```

