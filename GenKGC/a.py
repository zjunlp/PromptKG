from transformers import BartForConditionalGeneration, AutoTokenizer, T5ForConditionalGeneration

model_name_or_path = "google/t5-v1_1-base"
model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


from models.trie import Trie

label_sequence = ["I think it is terrible", "I think it is good"]
label_sequence_ids = tokenizer(label_sequence, add_special_tokens=True).input_ids
trie = Trie([[0] + _ for _ in label_sequence_ids])

print(label_sequence_ids)

inputs = tokenizer("I hate this movie", return_tensors='pt')
outputs = model.generate(**inputs, prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
    num_beams=5,
    num_return_sequences=3,
    output_scores=True
)
# outputs = model.generate(**inputs)
print(outputs)

