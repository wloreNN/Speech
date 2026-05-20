import librosa
import numpy as np
import matplotlib.pyplot as plt


def extract_pitch(y, sr):
    """
    SDD 6 (Algoritma 1): Perde Tespiti ve Takibi (Pitch Detection & Tracking).
    Duygu ve cinsiyet dönüşümü için temel frekansı (F0) çıkarır.
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    f0 = np.nan_to_num(f0)
    return f0


def extract_spectral_features(y, sr):
    """
    SDD 4.3 & 6: Spektral Zarf ve Formant Analizi için MFCC ve RMS.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rms = librosa.feature.rms(y=y)
    return mfccs, rms


def extract_speaker_embedding(y, sr):
    """
    SRS FR1.18 / SDD 6 (Algoritma 3): Konuşmacı Temsili Çıkarma.

    MFCC istatistiklerini (ortalama + std) ve ortalama F0'ı birleştirerek
    konuşmacı kimliğini temsil eden basit bir gömme vektörü üretir.
    Derin öğrenme gerektirmeden DSP tabanlı konuşmacı dönüşümü için kullanılır.

    Döndürür:
        dict:
            'mfcc_mean'  – MFCC kanallarının ortalama değerleri  (20,)
            'mfcc_std'   – MFCC kanallarının standart sapması     (20,)
            'mean_f0'    – Sesli çerçevelerin ortalama F0'ı (Hz)
    """
    mfccs = librosa.feature.mfcc(y=y.astype(np.float32), sr=sr, n_mfcc=20)

    f0, voiced_flag, _ = librosa.pyin(
        y.astype(np.float32),
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )

    if f0 is not None and np.any(voiced_flag):
        f0_voiced = f0[voiced_flag]
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
        mean_f0 = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 150.0
    else:
        mean_f0 = 150.0

    return {
        'mfcc_mean': np.mean(mfccs, axis=1),
        'mfcc_std': np.std(mfccs, axis=1),
        'mean_f0': mean_f0,
    }


def visualize_features(f0, title="Pitch Contour (F0)"):
    """
    SDD 4.4: Analiz sonuçlarını görselleştirir.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(f0, label='F0 (Hz)', color='r')
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Frekans (Hz)")
    plt.legend()
    plt.show()