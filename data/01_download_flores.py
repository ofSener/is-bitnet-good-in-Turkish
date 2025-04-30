from datasets import load_dataset, disable_caching

disable_caching()
tur = load_dataset(
    "facebook/flores",
    "tur_Latn",
    split="devtest",
    trust_remote_code=True
).select(range(2000))  # ilk 2000 cümle

tur.save_to_disk("data/tur_subset")
print("Türkçe alt küme kaydedildi.")
