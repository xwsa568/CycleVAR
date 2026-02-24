from transformers import AutoTokenizer, T5EncoderModel

save_dir = "./ckpts/flan-t5-xl"

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
text_encoder = T5EncoderModel.from_pretrained("google/flan-t5-xl")

tokenizer.save_pretrained(save_dir)
text_encoder.save_pretrained(save_dir)
