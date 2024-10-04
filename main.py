from nltk.stem import SnowballStemmer
from transformers import MarianMTModel, MarianTokenizer

stemmer = SnowballStemmer("english")

def preprocess_text(text):
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

def translate_text(text, src_lang="en", tgt_lang="ru"):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return translated_text

if __name__ == "__main__":
    input_text = input("Введите текст для перевода: ")
    src_lang = input("Введите исходный язык (en, fr, de и т.д.): ")
    tgt_lang = input("Введите целевой язык (ru, en, fr и т.д.): ")
    
    preprocessed_text = preprocess_text(input_text)
    print("Предобработанный текст:", preprocessed_text)

    translated_text = translate_text(preprocessed_text, src_lang=src_lang, tgt_lang=tgt_lang)
    print("Переведённый текст:", translated_text)