"""
SDD 4.4 / SRS FR1.24–FR1.25: Yardımcı araçlar.
Ses dışa aktarma, spektral eğim uygulama ve format dönüşümleri.
"""

import io
import numpy as np
import librosa
import soundfile as sf


def audio_to_bytes(y: np.ndarray, sr: int, fmt: str = "WAV") -> bytes:
    """
    NumPy ses dizisini belirli bir formatta bayt dizisine dönüştürür.
    Streamlit download_button ve diğer çıktı noktaları için kullanılır.

    Parametreler
    ------------
    y   : ses örnekleri (float, [-1, 1])
    sr  : örnekleme hızı
    fmt : 'WAV' veya 'FLAC'
    """
    buf = io.BytesIO()
    # SoundFile float32 yazarken [-1, 1] aralığını doğal kabul eder
    sf.write(buf, y.astype(np.float32), sr, format=fmt)
    buf.seek(0)
    return buf.read()


def apply_spectral_tilt(y: np.ndarray, sr: int, tilt_db: float = 0.0) -> np.ndarray:
    """
    SRS FR1.12: Spektral eğim (spectral tilt) uygulama.

    Pozitif tilt_db → yüksek frekanslara enerji katar (parlak/tiz ses).
    Negatif tilt_db → düşük frekanslara enerji katar (koyu/bas ses).

    Parametreler
    ------------
    y        : ses dizisi
    sr       : örnekleme hızı
    tilt_db  : uç frekansta uygulanacak toplam kazanç (dB)
    """
    if tilt_db == 0.0:
        return y

    D     = librosa.stft(y)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=(D.shape[0] - 1) * 2)

    # Frekans eksenine doğrusal dB eğimi
    tilt_profile = np.linspace(0.0, tilt_db, len(freqs))
    tilt_linear  = 10.0 ** (tilt_profile / 20.0)

    D_tilted = D * tilt_linear[:, np.newaxis]
    y_out    = librosa.istft(D_tilted, length=len(y))
    max_val  = np.max(np.abs(y_out))
    if max_val > 1.0:
        y_out = y_out / max_val
    return y_out


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """
    Sesi [-1, 1] aralığına normalize eder.
    """
    max_val = np.max(np.abs(y))
    return y / max_val if max_val > 0 else y


def resample_if_needed(y: np.ndarray, orig_sr: int, target_sr: int = 16000) -> tuple:
    """
    Gerekirse yeniden örnekler ve (y, target_sr) döndürür.
    """
    if orig_sr == target_sr:
        return y, orig_sr
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr), target_sr
