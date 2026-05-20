import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


def preprocess_audio(file_path, target_sr=16000, noise_reduce=False,
                     noise_strength=0.75, stationary_noise=False):
    """
    SRS FR1.1 – FR1.7: Ses yükleme, resample, normalizasyon ve sessizlik temizleme.
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=20)

    if noise_reduce:
        y = reduce_noise(y, sr, strength=noise_strength,
                         stationary=stationary_noise)
    return y, sr


def preprocess_audio_array(y, sr, target_sr=16000, noise_reduce=False,
                            noise_strength=0.75, stationary_noise=False):
    """
    Ham NumPy ses dizisi için preprocessing (mikrofon girişi için kullanılır).
    """
    # Mono'ya dönüştür
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Yeniden örnekle
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    y = librosa.util.normalize(y.astype(np.float32))
    y, _ = librosa.effects.trim(y, top_db=20)

    if noise_reduce:
        y = reduce_noise(y, sr, strength=noise_strength,
                         stationary=stationary_noise)
    return y, sr


def reduce_noise(y, sr, strength=0.75, stationary=False):
    """
    SRS FR1.5: Spectral Gating tabanlı gürültü azaltma (noisereduce).

    stationary=False → Ani/geçici gürültüler dahil her tür gürültüyü bastırır
                        (masa çarpma, tıklama, kısa sesler).
    stationary=True  → Sabit arka plan gürültüsüne (fan, uğultu) odaklanır,
                        daha hızlı çalışır.
    strength         → 0.0–1.0 arası; yüksek değer daha agresif azaltma.
    """
    import noisereduce as nr

    reduced = nr.reduce_noise(
        y=y.astype(np.float32),
        sr=sr,
        prop_decrease=float(strength),
        stationary=stationary,
        # Non-stationary modda kayan pencere
        time_constant_s=1.0 if not stationary else 2.0,
    )
    return librosa.util.normalize(reduced)

def visualize_audio(y, sr, title="Ses Sinyali"):
    """
    SDD 4.4 & 8: Çıktı görselleştirme ve spektrogram. [cite: 215, 219, 289]
    """
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.title(title)
    plt.xlabel("Zaman (s)")
    plt.ylabel("Genlik")
    plt.tight_layout()
    plt.show()

# --- TEST ---
# file_path = "ornek_ses.wav" # Buraya kendi dosya yolunuzu yazın
# audio, rate = preprocess_audio(file_path)
# visualize_audio(audio, rate, title="Ön İşlemeden Geçmiş Ses")