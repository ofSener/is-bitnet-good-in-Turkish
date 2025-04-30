#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
from tqdm import tqdm  # İlerleme çubuğu için
import torch
from datasets import load_from_disk
# Gerekli Transformers sınıflarını import et
# LlamaTokenizer'ı da özellikle import ediyoruz
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

# --- ❶ Sabitler ---
# Kullanılacak modelin Hugging Face ID'si
MODEL_ID = "1bitLLM/bitnet_b1_58-xl"
# Veri setinin bulunduğu dizin (script dosyasının konumuna göre)
# Bu script 'scripts' klasöründeyse, veri 'data' klasöründe olmalı
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/tur_subset"))
# Belleğe bağlı olarak ayarlanabilecek batch boyutu
BATCH_SIZE = 8
# Kullanılacak veri tipi (GPU desteğine ve istenen hassasiyete bağlı)
# Örn: torch.bfloat16, torch.float16, torch.float32
TORCH_DTYPE = torch.bfloat16

# --- ❷ Adım 1: Veri yükleme ---
print("[1/4] Yükleniyor: Türkçe cümleler…")
try:
    # Veri setini diskten yükle ve 'sentence' sütununu al
    texts = load_from_disk(DATA_DIR)["sentence"]
    print(f"[+] {len(texts)} cümle yüklendi.\n")
# Hata durumlarını yakala
except FileNotFoundError:
    print(f"[HATA] Veri dizini bulunamadı: {DATA_DIR}")
    print("Lütfen DATA_DIR değişkeninin doğru 'tur_subset' dizinini gösterdiğinden emin olun.")
    sys.exit(1)  # Hata durumunda programdan çık
except Exception as e:
    print(f"[HATA] Veri yüklenirken bir hata oluştu: {e}")
    sys.exit(1)

# --- ❸ Adım 2: Tokenizer & Model yükleme ---
print("[2/4] Yükleniyor: Tokenizer ve model…")
try:
    # --- GÜNCELLENMİŞ TOKENIZER YÜKLEME ---
    # AutoTokenizer bu model için sorun çıkardığından, tokenizer_config.json'da
    # belirtilen LlamaTokenizer sınıfını DOĞRUDAN yüklüyoruz.
    print(f"Tokenizer ({MODEL_ID}) için LlamaTokenizer doğrudan yükleniyor...")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID) # Doğrudan LlamaTokenizer kullan
    print("LlamaTokenizer başarıyla yüklendi.")
    # --- GÜNCELLENMİŞ TOKENIZER YÜKLEME SONU ---

    # Modeli yükle. Modelin özel kodu olduğu için trust_remote_code=True GEREKLİ.
    print("Model yükleniyor...")
    # Cihazı belirle (varsa GPU kullan)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model için kullanılacak cihaz: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,  # Modelin özel kodu için GEREKLİ
        torch_dtype=TORCH_DTYPE  # Belirlenen veri tipini kullan
    ).to(device).eval()  # Modeli uygun cihaza taşı ve değerlendirme moduna al
    print("[+] Model ve tokenizer hazır.\n")

# Olası hataları yakala
except ImportError as e:
    print(f"[HATA] Gerekli kütüphane bulunamadı: {e}")
    print("Lütfen 'transformers', 'torch', 'datasets', 'tqdm' kütüphanelerinin sanal ortamınızda kurulu olduğundan emin olun.")
    sys.exit(1)
except ValueError as e:
    # ValueError özellikle model/tokenizer yükleme hatalarında faydalı olabilir
    print(f"[HATA] Model/Tokenizer yüklenirken değer hatası: {e}")
    print(f"Model ID ({MODEL_ID}) veya Hugging Face üzerindeki yapılandırması ile ilgili bir sorun olabilir.")
    sys.exit(1)
except Exception as e:
    print(f"[HATA] Model/Tokenizer yüklenirken genel hata: {e}")
    import traceback
    traceback.print_exc() # Detaylı hata ayıklama için traceback yazdır
    sys.exit(1)

# --- ❹ Adım 3: Perplexity hesaplama ---
print(f"[3/4] Başlatılıyor: Perplexity hesaplama (batch_size={BATCH_SIZE})...")
total_neg_log_likelihood = 0.0  # Toplam negatif log olabilirlik
total_tokens = 0                # Toplam işlenen (padding olmayan) token sayısı

# Veriyi batch'ler halinde işle
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Batch"):
    batch = texts[i : i + BATCH_SIZE]  # Mevcut batch'i al
    try:
        # Batch'i tokenize et. Padding ekle, PyTorch tensörleri döndür.
        # Güvenlik için truncation ve modelin maksimum uzunluğunu ekle.
        # Modelin max_position_embeddings değerini kullanmak daha dinamik olabilir
        # ancak 512 genellikle güvenli bir varsayılandır.
        max_length = 512 
        # try:
        #    max_length = model.config.max_position_embeddings
        # except AttributeError:
        #    max_length = 512 # Eğer model config'de yoksa varsayılan kullan
            
        encodings = tokenizer(
            batch,
            return_tensors="pt",        # PyTorch tensörleri olarak döndür
            padding=True,               # En uzun cümleye göre padding ekle
            truncation=True,            # Cümleleri maksimum uzunluğa göre kırp
            max_length=max_length       # Belirlenen maksimum uzunluk
        )
        # Tensörleri modele gönderilecek cihaza taşı
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        # ÖNEMLİ DÜZELTME: Kayıp hesaplamasında padding token'larını ignore et.
        # Etiketleri (labels) oluştururken padding olan yerlere -100 ata.
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100 # Padding token maskesi == 0 olan yerler

        # Gradyan hesaplaması yapmadan modeli çalıştır (inference)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels  # Padding tokenları -100 yapılmış etiketleri kullan
            )

            # outputs.loss, batch'deki padding olmayan token başına *ortalama* kaybı verir.
            # Toplam negatif log olasılığı bulmak için, bu ortalama kaybı
            # batch'deki padding olmayan token sayısıyla çarpmamız gerekir.
            num_tokens_in_batch = (labels != -100).sum().item()

            if num_tokens_in_batch > 0:
               # Batch'in toplam negatif log olasılığı
               batch_neg_log_likelihood = outputs.loss.item() * num_tokens_in_batch
               # Genel toplama ekle
               total_neg_log_likelihood += batch_neg_log_likelihood
               total_tokens += num_tokens_in_batch
            # else: Bu batch sadece padding içeriyorsa (çok kısa cümlelerde olabilir), atla.

    except Exception as e:
        # Batch işleme sırasında hata olursa bildir ve isteğe bağlı olarak devam et/çık
        print(f"\n[HATA] Batch {i//BATCH_SIZE + 1} işlenirken hata: {e}")
        print(f"Batch içeriği (ilk 50 karakter): {[text[:50] + '...' for text in batch]}")
        # Hata durumunda programdan çıkmayı veya sadece bu batch'i atlamayı seçebilirsiniz
        # continue # Bu satırı aktif ederseniz hatalı batch atlanır
        sys.exit(1) # Programdan çık

# --- Perplexity Hesaplaması ---
# Son perplexity değerini hesapla
if total_tokens == 0:
    print("[HATA] Hiç geçerli token işlenemedi. Perplexity hesaplanamıyor.")
    perplexity = float('inf') # Sonsuz olarak ata veya hata ver
else:
    # Perplexity = exp( toplam_negatif_log_olasilik / toplam_token_sayisi )
    perplexity = math.exp(total_neg_log_likelihood / total_tokens)

print("\n[+] Hesaplama tamamlandı.\n")

# --- ❺ Adım 4: Sonuç ---
# Hesaplanan perplexity değerini yazdır (hassasiyeti biraz artırıldı)
print(f"[4/4] Türkçe Perplexity ({total_tokens} token üzerinden): {perplexity:.4f}")