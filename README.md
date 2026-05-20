# 〰 VoiceMorph Pro (Speech Warping Project)

Bu proje, Dijital Sinyal şleme (DSP) teknikleri ve **PyWorld Vocoder** kullanılarak geliştirilmiş gelişmiş bir ses dönüştürme (Speech Warping) ve analiz sistemidir. Gelişmiş kullanıcı arayüzü **Streamlit** ile tasarlanmıştır.

## ✨ Özellikler

- **😊 Duygu Dönüşümü (Emotion Conversion):** Sesin temposunu, perdesini (pitch), enerjisini ve spektral eğimini manipüle ederek sese "Üzgün, Sinirli, Heyecanlı, Sakin, Fısıltı" gibi duygular katar. Serbest mod ile özel kaydırıcıları (slider) kullanarak ince ayar yapabilirsiniz.
- **👤 Cinsiyet ve Yaş Dönüşümü:** Pitch Shift (St) ve Formant Shift (Ses yolu çarpanı) özelliklerini kullanarak sesleri Erkekten Kadına, Kadından Erkeğe veya Yetişkinden Çocuğa dönüştürür.
- **📊 Performans Metrikleri & Analiz:** Gerçek Zamanlılık Faktörü (RTF) ve Sinyal-Gürültü Oranı (SNR) hesaplar. Ayrıca girdi seslerine ait Temel Frekans (F0) ve Mel-Frekans Spektrogramı (MFCC) grafiklerini çizer.
- **🎨 Modern Arayüz:** Karanlık tema (Dark Mode) destekli, kolay kullanımlı modüler arayüz.

---

## 🛠️ Kurulum Gereksinimleri

Projenin bilgisayarınızda sorunsuz çalışabilmesi için **Python 3.8 veya üzeri** bir sürümün yüklü olması önerilir.

### 1. Ortamı Hazırlama
Projeyi indirdikten sonra proje ana dizininde bir sanal ortam (virtual environment) oluşturmanız tavsiye edilir:

```bash
# Sanal ortam oluşturma
python -m venv .venv

# Sanal ortamı aktifleştirme (Windows)
.\.venv\Scripts\activate

# Sanal ortamı aktifleştirme (MacOS/Linux)
source .venv/bin/activate
```

### 2. Gerekli Kütüphaneleri Yükleme
Sistem için gerekli olan Ses şleme (`librosa`, `pyworld`) ve Arayüz (`streamlit`) gibi paketleri kurun:

```bash
pip install -r requirements.txt
```

*(Not: `pyworld` kütüphanesinin kurulumu sırasında sisteminizde C++ Build Tools - Visual Studio Derleme Araçları'na ihtiyaç duyulabilir.)*

---

## 🚀 Uygulamayı Çalıştırma

Tüm kurulumlar tamamlandıktan sonra, terminalde/komut satırında aktif sanal ortamınız içindeyken uygulamayı başlatmak için şu komutu girin:

```bash
streamlit run app.py
```

Bu komutu girdikten sonra tarayıcınız otomatik olarak açılacak ve `http://localhost:8501` adresinde uygulamanın arayüzü karşınıza gelecektir.

## 📂 Proje Yapısı

- `app.py`: Streamlit tabanlı ana web arayüzü ve entegrasyon dosyası.
- `src/`: Sinyal işleme modüllerinin (Backend) bulunduğu klasör.
  - `preprocessing.py`: Ses yükleme, sessizlik kırpma ve normalizasyon.
  - `features.py`: F0 (Pitch) çıkarma ve MFCC analizi.
  - `transformation.py`: PyWorld Vocoder üzerinden Formant, Zaman ve Pitch kaydırma motoru.
  - `evaluation.py`: Sinyal SNR ve RTF hesaplama standartları.
- `.streamlit/`: Özel tema konfigürasyon dosyaları.
