# from transformers import LukeTokenizer, LukeForEntityPairClassification
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer


model = AutoModelForTokenClassification.from_pretrained("studio-ousia/luke-base")
tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base")
ner_model = pipeline('ner', model=model, tokenizer=tokenizer)
text = "Beyoncé lives in Los Angeles."
print(ner_model(text))


# model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
# tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")

# entity_spans = [(0, 7), (17, 28)]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"
# inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits
# predicted_class_idx = int(logits[0].argmax())
# print("Predicted class:", model.config.id2label[predicted_class_idx])