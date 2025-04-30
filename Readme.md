# Türkçe Dil Modeli Değerlendirme (BitNet) | Turkish Language Model Evaluation (BitNet)

Bu proje, Hugging Face üzerinde bulunan `1bitLLM/bitnet_b1_58-xl` modelinin Türkçe metinler üzerindeki performansını Perplexity ve BLEU skorları kullanarak değerlendirmeyi amaçlamaktadır.

This project aims to evaluate the performance of the `1bitLLM/bitnet_b1_58-xl` model from Hugging Face on Turkish text using Perplexity and BLEU scores.

Proje, özellikle deneysel modellerle çalışırken karşılaşılabilecek kurulum ve yapılandırma zorluklarını ve bunların çözüm adımlarını da belgelemektedir.

The project also documents the setup and configuration challenges encountered, particularly when working with experimental models, and the steps taken to resolve them.

---

## Proje Yapısı | Project Structure

```plaintext
bitnet_turkce/
├── data/
│   ├── tur_subset/            # Değerlendirme için kullanılan Türkçe alt küme verisi (Dataset)
│   └── 01_download_flores.py  # (Varsa) Veri indirme betiği (Data download script, if applicable)
├── scripts/
│   ├── 02_compute_perplexity.py # Perplexity hesaplama betiği (Perplexity calculation script)
│   └── 03_compute_bleu.py     # BLEU hesaplama betiği (BLEU calculation script)
├── venv/                    # Python sanal ortamı (Python virtual environment)
├── .gitignore               # Git tarafından yok sayılacak dosyalar (Files ignored by Git)
├── requirements.txt         # Gerekli Python kütüphaneleri (Required Python libraries)
└── README.md                # Bu dosya (This file)
```

---

## Kurulum | Setup

Projeyi çalıştırmak için aşağıdaki adımları izleyin: / Follow these steps to run the project:

1.  **Ön Gereksinimler | Prerequisites:**
    * `git`
    * `git-lfs`: (Gerekliyse / If needed for large files)
        ```bash
        # Ubuntu/Debian:
        sudo apt update && sudo apt install git-lfs
        # Kurulumdan sonra / After installation:
        git lfs install --system
        ```
    * `python` (>= 3.10 önerilir / recommended)

2.  **Projeyi Klonlama | Clone the Project:**
    ```bash
    git clone <proje_repo_url>
    cd <proje_dizini>
    ```

3.  **Sanal Ortam | Virtual Environment:**
    ```bash
    # Sanal ortam oluştur / Create virtual environment
    python3 -m venv venv 
    # Sanal ortamı aktive et / Activate virtual environment
    # Linux/macOS:
    source venv/bin/activate
    # Windows (Git Bash/WSL):
    # source venv/Scripts/activate 
    # Windows (CMD/PowerShell):
    # .\venv\Scripts\activate 
    ```

4.  **Kütüphaneleri Kurma | Install Libraries:**
    * Proje ana dizininde aşağıdaki içeriğe sahip bir `requirements.txt` dosyası olduğundan emin olun: / Ensure you have a `requirements.txt` file in the project root with the following content:
        ```txt
        transformers>=4.40.0
        datasets>=2.18.0
        evaluate>=0.4.2
        torch>=2.2.0
        git+[https://github.com/huggingface/huggingface_hub.git](https://github.com/huggingface/huggingface_hub.git)
        tqdm
        protobuf
        ```
    * Kütüphaneleri kurun (Tam yol kullanılması önceki sorunlar nedeniyle tavsiye edilir): / Install the libraries (Using the full path is recommended due to previous issues):
        ```bash
        # Önce pip'i güncelle / Update pip first
        ./venv/bin/python3 -m pip install --upgrade pip setuptools
        # Gereksinimleri kur / Install requirements
        ./venv/bin/python3 -m pip install -r requirements.txt
        ```

---

## Kullanım | Usage

Kurulum tamamlandıktan sonra aşağıdaki scriptleri çalıştırabilirsiniz: / After setup, you can run the following scripts:

1.  **Veri Hazırlama (Gerekirse) | Data Preparation (If needed):**
    Eğer `data/tur_subset` dizini boşsa ve `data/01_download_flores.py` scripti veriyi indiriyorsa: / If the `data/tur_subset` directory is empty and the `data/01_download_flores.py` script downloads the data:
    ```bash
    ./venv/bin/python3 data/01_download_flores.py
    ```

2.  **Perplexity Hesaplama | Calculate Perplexity:**
    ```bash
    ./venv/bin/python3 scripts/02_compute_perplexity.py
    ```
    *Örnek Çıktı / Example Output:*
    ```
    [1/4] Yükleniyor: Türkçe cümleler…
    [+] 200 cümle yüklendi.
    ...
    [4/4] Türkçe Perplexity (13511 token üzerinden): 168127.5893 
    ```

3.  **BLEU Hesaplama | Calculate BLEU:**
    ```bash
    ./venv/bin/python3 scripts/03_compute_bleu.py
    ```
    *Bu script, modelin referans cümlelere göre tahminler üretmesini ve BLEU skorunu hesaplamasını sağlar. / This script generates predictions based on reference sentences and calculates the BLEU score.*

---

## Karşılaşılan Zorluklar ve Çözümler | Troubleshooting & Key Findings

Bu projede, özellikle `1bitLLM/bitnet_b1_58-xl` modeliyle çalışırken bazı zorluklarla karşılaşıldı: / Several challenges were encountered while working with the `1bitLLM/bitnet_b1_58-xl` model:

1.  **Sanal Ortam PATH Sorunları | Virtual Environment PATH Issues:** `pip` ve `python3` komutları, aktif sanal ortama rağmen global kurulumları işaret ediyordu. / `pip` and `python3` commands pointed to global installations despite the venv being active.
    * **Çözüm | Solution:** Komutlar, sanal ortam içindeki çalıştırılabilir dosyaların tam yolu belirtilerek çalıştırıldı (`./venv/bin/python3 ...`). / Commands were executed using the full path to the executables within the venv (`./venv/bin/python3 ...`).

2.  **Bozuk `pip` Kurulumu | Corrupted `pip` Installation:** Sanal ortamdaki `pip`, `pip._vendor.packaging` hatası vererek bozuldu. / `pip` within the venv became corrupted, raising a `pip._vendor.packaging` error.
    * **Çözüm | Solution:** Sanal ortam silinip yeniden oluşturuldu. / The virtual environment was deleted and recreated.

3.  **`1bitLLM/bitnet_b1_58-xl` Tokenizer Sorunu | Tokenizer Issue:** Bu model için tokenizer yüklemesi problemliydi. `AutoTokenizer` (hem `trust_remote_code=True` ile hem de olmadan) var olmayan `BitnetTokenizer` sınıfını arama hatası verdi. Modelin kendi `tokenizer_config.json` dosyası `LlamaTokenizer` belirtmesine rağmen bu sorun yaşandı. / Loading the tokenizer for this model was problematic. `AutoTokenizer` (both with and without `trust_remote_code=True`) failed, trying to find a non-existent `BitnetTokenizer` class, despite the model's `tokenizer_config.json` specifying `LlamaTokenizer`.
    * **Çözüm | Solution:** `LlamaTokenizer` sınıfı doğrudan yüklendi (`LlamaTokenizer.from_pretrained(MODEL_ID)`). / The `LlamaTokenizer` class was loaded directly (`LlamaTokenizer.from_pretrained(MODEL_ID)`).
    * **Kalıntı Uyarı | Lingering Warning:** Bu çözüm çalışsa da, yükleme sırasında hala sınıf uyumsuzluğu uyarısı ("...checkpoint is 'BitnetTokenizer'. The class this function is called from is 'LlamaTokenizer'.") alınıyor. Bu durum şimdilik scriptin çalışmasını engellemedi. / Although this worked, a class mismatch warning ("...checkpoint is 'BitnetTokenizer'. The class this function is called from is 'LlamaTokenizer'.") still appears during loading but hasn't prevented execution so far.

4.  **Eksik `protobuf` Kütüphanesi | Missing `protobuf` Library:** Model yüklemesi için `protobuf` kütüphanesi gerekti. / The `protobuf` library was required for model loading.
    * **Çözüm | Solution:** Kütüphane sanal ortama kuruldu (`pip install protobuf`). / The library was installed into the venv (`pip install protobuf`).

---

## Sonuçlar (Ön) | Preliminary Results

1.  **Perplexity:**
    * **Model:** `1bitLLM/bitnet_b1_58-xl`
    * **Veri | Data:** `tur_subset` (200 cümle / sentences)
    * **Uyarı | Warning:** Tokenizer sınıfı uyumsuzluk uyarısı alındı. / Tokenizer class mismatch warning was observed.
    * **Skor | Score:** `168127.59` (13511 token üzerinden / over 13511 tokens)
    * **Yorum | Interpretation:** Perplexity skoru aşırı yüksektir. Bu, modelin bu Türkçe veri setini modellemede çok başarısız olduğunu veya altta yatan sorunların (tokenizer uyumsuzluğu, model kalitesi) skoru ciddi şekilde etkilediğini göstermektedir. / The perplexity score is extremely high, indicating the model performs very poorly on this Turkish dataset, possibly due to model suitability issues, data mismatch, or the tokenizer inconsistency.

2.  **BLEU:**
    * **Durum | Status:** Hesaplama devam ediyor / devam ediyordu (README yazıldığı sırada). Sonuçlar elde edildiğinde buraya eklenecektir. / Calculation was in progress when this README was written. Results will be added here once available.

---