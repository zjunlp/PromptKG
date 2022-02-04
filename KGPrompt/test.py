from transformers import BartForConditionalGeneration, BartTokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", force_bos_token_to_be_generated=True)
tok = BartTokenizer.from_pretrained("facebook/bart-large")
example_english_phrase = "UN Chief Says There Is No <mask> in Syria"
batch = tok(example_english_phrase, return_tensors='pt')
generated_ids = model.generate(batch['input_ids'])
assert tok.batch_decode(generated_ids, skip_special_tokens=True) == ['UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria']