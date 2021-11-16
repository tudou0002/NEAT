For fine-tuning a model for NER, copy the train, test and dev sets in a separate folder and name them `train.txt`, `test.txt` and `val.txt` and run

```
python run_ner.py  --data_dir /path/to/folder/containing/train/and/test   --model_type bert   --labels path/to/labels/text/file  --model_name_or_path 'bert-base-cased'  --output_dir /path/to/output/directory  --max_seq_length  512   --num_train_epochs 10  --per_gpu_train_batch_size 8   --save_steps 500   --logging_steps 100   --seed 42  --do_train --do_predict --overwrite_output_dir --overwrite_cache --learning_rate 5e-5
```