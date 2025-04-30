#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys  # Hata durumunda çıkış yapmak için
from tqdm import tqdm
import torch  # Cihaz yönetimi ve veri tipi için
from datasets import load_from_disk
# LlamaTokenizer'ı özellikle import ediyoruz
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from evaluate import load  # BLEU metriğini yüklemek için

# --- ❶ Sabitler ---
MODEL_ID = "1bitLLM/bitnet_b1_58-xl"  # Kullanılacak model ID'si
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/tur_subset")) # Veri dizini
TORCH_DTYPE = torch.bfloat16 # Model için veri tipi
MAX_NEW_TOKENS = 50  # Her cümle için üretilecek maksimum yeni token sayısı

# --- ❷ Adım 1: Tokenizer & Model yükleme ---
print("[1/3] Yükleniyor: Tokenizer ve model…")
try:
    # Önceki scriptte işe yarayan yöntem: LlamaTokenizer'ı doğrudan yükle
    print(f"Tokenizer ({MODEL_ID}) için LlamaTokenizer doğrudan yükleniyor...")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
    print("LlamaTokenizer başarıyla yüklendi.")

    # Modeli yükle (trust_remote_code=True gerekli)
    print("Model yükleniyor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model için kullanılacak cihaz: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,  # Özel model kodu için gerekli
        torch_dtype=TORCH_DTYPE
    ).to(device).eval()  # Cihaza taşı ve değerlendirme moduna al
    print("[+] Model ve tokenizer hazır.\n")

# Olası import ve yükleme hatalarını yakala
except ImportError as e:
    print(f"[HATA] Gerekli kütüphane bulunamadı: {e}")
    print("Lütfen 'transformers', 'torch', 'datasets', 'evaluate', 'protobuf', 'tqdm' kütüphanelerinin kurulu olduğundan emin olun.")
    sys.exit(1)
except Exception as e:
    print(f"[HATA] Model/Tokenizer yüklenirken genel hata: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- ❸ Adım 2: Türkçe cümleler (Referanslar) yükleme ---
print("[2/3] Yükleniyor: Türkçe cümleler (referanslar)…")
try:
    # Veri setinden cümleleri yükle (bunlar BLEU için referans olacak)
    references_raw = load_from_disk(DATA_DIR)["sentence"]
    print(f"[+] {len(references_raw)} cümle (referans) yüklendi.\n")
# Hata durumlarını yakala
except FileNotFoundError:
    print(f"[HATA] Veri dizini bulunamadı: {DATA_DIR}")
    sys.exit(1)
except Exception as e:
    print(f"[HATA] Veri yüklenirken bir hata oluştu: {e}")
    sys.exit(1)


# --- ❹ Adım 3: Tahmin üretme ve BLEU hesaplama ---
print("[3/3] Başlatılıyor: Tahmin üretme ve BLEU hesaplama…")
predictions = [] # Modelin üreteceği tahminleri saklamak için liste
try:
    # Referans cümleler üzerinden tek tek geçerek tahmin üret
    for sentence in tqdm(references_raw, desc="Üretiliyor"):
        # Giriş cümlesini tokenize et ve modelin bulunduğu cihaza gönder
        inputs = tokenizer(sentence, return_tensors="pt").to(device)

        # Gradyan hesaplaması yapmadan token üret (inference için daha verimli)
        with torch.no_grad():
             output_ids = model.generate(
                 **inputs, # Tokenize edilmiş girdiyi modele ver
                 max_new_tokens=MAX_NEW_TOKENS, # Maksimum yeni token sayısı
                 # Not: Daha iyi sonuçlar için do_sample=True, temperature, top_k gibi
                 #      parametreler eklenebilir, ancak BLEU için genellikle greedy search yeterlidir.
                 # pad_token_id=tokenizer.eos_token_id # Gerekli olabilir
             )

        # Üretilen tokenları decode et.
        # ÖNEMLİ DÜZELTME: model.generate girdi aldığında, çıktıda girdi tokenlarını da verir.
        # Sadece yeni üretilen kısmı decode etmeliyiz.
        input_length = inputs["input_ids"].shape[1] # Girdi token sayısı
        # Sadece input_length'den sonraki tokenları decode et
        decoded_output = tokenizer.decode(output_ids[0, input_length:], skip_special_tokens=True)
        predictions.append(decoded_output) # Tahmini listeye ekle

# Üretim sırasında oluşabilecek hataları yakala
except Exception as e:
    print(f"\n[HATA] Tahmin üretimi sırasında hata: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# BLEU metriğini yükle
print("BLEU metriği yükleniyor...")
try:
    bleu = load("bleu")
except FileNotFoundError:
     print("[HATA] BLEU metriği yüklenemedi. İnternet bağlantınızı kontrol edin veya evaluate cache'ini gözden geçirin.")
     sys.exit(1)
except Exception as e:
    print(f"[HATA] BLEU metriği yüklenirken hata: {e}")
    sys.exit(1)


# BLEU hesaplaması için referansları ve tahminleri hazırla
# Referanslar: List[List[List[str]]] formatında olmalı (Her cümle için bir referans listesi, her referans kelime listesi)
# Tahminler: List[List[str]] formatında olmalı (Her cümle için bir kelime listesi)
references_for_bleu = [[ref.split()] for ref in references_raw]
predictions_for_bleu = [p.split() for p in predictions]

print("BLEU hesaplanıyor...")
try:
    # BLEU skorunu hesapla
    result = bleu.compute(predictions=predictions_for_bleu, references=references_for_bleu)
    # BLEU skoru genelde 0-1 arasındadır, 100 ile çarparak yüzde olarak gösterelim
    bleu_score = result['bleu'] * 100
    print(f"\n[+] Türkçe BLEU: {bleu_score:.2f}")
except Exception as e:
    print(f"[HATA] BLEU hesaplanırken hata: {e}")
    # Hata ayıklamaya yardımcı olmak için bazı verileri yazdır
    print("Referans örneği (ilk):", references_for_bleu[0] if references_for_bleu else "Yok")
    print("Tahmin örneği (ilk):", predictions_for_bleu[0] if predictions_for_bleu else "Yok")
    sys.exit(1)