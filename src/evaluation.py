"""
SDD 4.5: Değerlendirme Modülü.
SNR, RTF, enerji oranı ve pitch sapması metriklerini hesaplar.
"""

import numpy as np
import librosa


def calculate_rtf(processing_time: float, audio_duration: float) -> float:
    """
    SDD 4.5 – RTF (Real-Time Factor): işleme süresinin ses süresine oranı.
    RTF < 1 gerçek zamanlı işlemeye yaklaşıldığını gösterir.
    """
    if audio_duration == 0:
        return 0.0
    return processing_time / audio_duration


def calculate_snr(original_signal: np.ndarray,
                  processed_signal: np.ndarray) -> float:
    """
    SDD 4.5 – SNR (Signal-to-Noise Ratio): çıkış sinyalinin kalitesini ölçer.
    Daha yüksek değer → daha az bozulma.
    """
    min_len = min(len(original_signal), len(processed_signal))
    orig = original_signal[:min_len]
    proc = processed_signal[:min_len]

    noise = orig - proc
    noise_power  = np.mean(noise ** 2)
    signal_power = np.mean(orig ** 2)

    if noise_power == 0:
        return float('inf')
    return 10.0 * np.log10(signal_power / noise_power)


def calculate_energy_ratio(original_signal: np.ndarray,
                            processed_signal: np.ndarray) -> float:
    """
    İşlenen sesin orijinal sese göre enerji değişim oranını hesaplar.
    1.0 = enerji değişmedi, >1 = güçlendi, <1 = zayıfladı.
    """
    orig_rms = np.sqrt(np.mean(original_signal ** 2))
    proc_rms = np.sqrt(np.mean(processed_signal ** 2))
    if orig_rms == 0:
        return 0.0
    return proc_rms / orig_rms


def calculate_pitch_deviation(original_signal: np.ndarray,
                               processed_signal: np.ndarray,
                               sr: int = 16000) -> float:
    """
    Orijinal ve işlenmiş ses arasındaki ortalama F0 sapmasını (yarım ton, st) döndürür.
    Pitch dönüşümlerinin doğruluğunu değerlendirmek için kullanılır.
    """
    def _mean_f0(y):
        f0, voiced, _ = librosa.pyin(
            y.astype(np.float32),
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
        )
        if f0 is None or not np.any(voiced):
            return None
        voiced_f0 = f0[voiced]
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
        return float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else None

    f0_orig = _mean_f0(original_signal)
    f0_proc = _mean_f0(processed_signal)

    if f0_orig is None or f0_proc is None or f0_orig == 0:
        return 0.0

    # Yarım ton cinsinden fark: 12 * log2(f_proc / f_orig)
    semitone_diff = 12.0 * np.log2(f0_proc / f0_orig)
    return round(semitone_diff, 2)

