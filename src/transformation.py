import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d


def shift_formants(sp, sr, env_ratio):
    """
    Spektral zarfı (formantları) 'env_ratio' oranında kaydırır.
    """
    if env_ratio == 1.0:
        return sp

    num_bins = sp.shape[1]
    freqs = np.linspace(0, sr // 2, num_bins)
    sp_shifted = np.zeros_like(sp)

    for i in range(sp.shape[0]):
        f_interp = interp1d(freqs, sp[i, :], kind='linear',
                            bounds_error=False, fill_value=0.0)
        target_freqs = freqs / env_ratio
        sp_shifted[i, :] = f_interp(target_freqs)

    return sp_shifted


def change_gender_age(y, sr, f0_ratio=1.0, env_ratio=1.0):
    """
    SRS FR1.13 – FR1.16: Cinsiyet ve Yaş Dönüşümü.
    PyWorld Vocoder ile Pitch (F0) ve ses yolu (formant) değiştirilir.
    """
    y = y.astype(np.float64)
    _f0, t = pw.dio(y, sr)
    f0 = pw.stonemask(y, _f0, t, sr)
    sp = pw.cheaptrick(y, f0, t, sr)
    ap = pw.d4c(y, f0, t, sr)

    modified_f0 = f0 * f0_ratio
    modified_sp = shift_formants(sp, sr, env_ratio)

    synthesized = pw.synthesize(modified_f0, modified_sp, ap, sr)
    synthesized = np.nan_to_num(synthesized, nan=0.0, posinf=0.0, neginf=0.0)
    return synthesized


def change_emotion(y, sr, emotion="neutral",
                   pitch_shift_st=None, speed_ratio=None, energy_scale=None):
    """
    SRS FR1.8 – FR1.12: Duygu Dönüşümü.
    Pitch, tempo ve enerjiyi manipüle eder.
    """
    y = np.nan_to_num(y.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    emo_params = {
        "neutral":   (1.0,  1.0,  1.0),
        "sad":       (0.8,  0.8,  0.6),
        "angry":     (1.3,  1.25, 1.4),
        "excited":   (1.4,  1.2,  1.3),
        "calm":      (0.9,  0.85, 0.7),
        "whispered": (0.0,  0.9,  1.0),
        "manual":    (1.0,  1.0,  1.0),
    }

    b_f0, b_speed, b_energy = emo_params.get(emotion, (1.0, 1.0, 1.0))

    final_f0_ratio = 2.0 ** (pitch_shift_st / 12.0) if pitch_shift_st is not None else b_f0
    final_speed    = speed_ratio if speed_ratio is not None else b_speed
    final_energy   = (energy_scale / 100.0) if energy_scale is not None else b_energy

    if final_speed != 1.0:
        y = librosa.effects.time_stretch(y, rate=final_speed)

    _f0, t = pw.dio(y, sr)
    f0 = pw.stonemask(y, _f0, t, sr)
    sp = pw.cheaptrick(y, f0, t, sr)
    ap = pw.d4c(y, f0, t, sr)

    if emotion == "whispered":
        modified_f0 = np.zeros_like(f0)
        ap = np.ones_like(ap)           # tamamen aperiyodik → fısıltı
    else:
        modified_f0 = f0 * final_f0_ratio

    synthesized = pw.synthesize(modified_f0, sp, ap, sr)
    synthesized = synthesized * final_energy
    return synthesized


# ─────────────────────────────────────────────────────────────────────────────
# KONUŞMACI DÖNÜŞÜMÜ  (SRS FR1.17 – FR1.20 / SDD Algoritma 3)
# ─────────────────────────────────────────────────────────────────────────────

def convert_speaker(y_src, sr, y_ref=None, sr_ref=None):
    """
    SRS FR1.17 – FR1.20: Konuşmacı Kimliği Dönüşümü.

    Kaynak konuşmacının F0 ortalaması ve MFCC spektral istatistikleri
    referans konuşmacıya normalize edilir; PyWorld ile yeniden sentezlenir.

    Parametreler
    ------------
    y_src   : kaynak ses dizisi
    sr      : kaynak örnekleme hızı
    y_ref   : referans konuşmacı sesi (None ise ön ayarlı örnek kullanılır)
    sr_ref  : referans ses örnekleme hızı
    """
    from src.features import extract_speaker_embedding   # geç içe aktarım – döngüsel bağımlılığı önler

    # Referans ses yoksa nötr dönüşüm yap
    if y_ref is None or len(y_ref) == 0:
        return y_src.copy()

    # Referansı kaynak SR'ye yeniden örnekle
    if sr_ref is not None and sr_ref != sr:
        y_ref = librosa.resample(y_ref, orig_sr=sr_ref, target_sr=sr)

    src_emb = extract_speaker_embedding(y_src.astype(np.float32), sr)
    ref_emb = extract_speaker_embedding(y_ref.astype(np.float32), sr)

    # F0 oranı: referans / kaynak ortalama pitch
    src_f0  = max(src_emb['mean_f0'], 50.0)
    ref_f0  = max(ref_emb['mean_f0'], 50.0)
    f0_ratio = np.clip(ref_f0 / src_f0, 0.5, 2.0)

    # Formant oranı: MFCC ortalamaları farkından tahmin
    src_mfcc_mean = np.mean(src_emb['mfcc_mean'])
    ref_mfcc_mean = np.mean(ref_emb['mfcc_mean'])
    mfcc_ratio = (ref_mfcc_mean / src_mfcc_mean) if abs(src_mfcc_mean) > 1e-6 else 1.0
    env_ratio  = np.clip(1.0 + (mfcc_ratio - 1.0) * 0.4, 0.65, 1.55)

    return change_gender_age(y_src, sr, f0_ratio=f0_ratio, env_ratio=env_ratio)


# ─────────────────────────────────────────────────────────────────────────────
# ŞARKI SESİ ÜRETME  (SRS FR1.21 – FR1.23 / SDD Algoritma 4)
# ─────────────────────────────────────────────────────────────────────────────

# Do majör gamının temel nota frekansları (Hz)
_C_MAJOR_SCALE = [261.63, 293.66, 329.63, 349.23,
                  392.00, 440.00, 493.88, 523.25]

_NOTE_MAP = {
    'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
    'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
    'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88,
}


def _parse_note_string(note_str):
    """
    'C4', 'A#3', 'D5' gibi nota adlarını Hz değerine çevirir.
    Octave numarası belirtilmişse frekansı ölçeklendirir.
    """
    note_str = note_str.strip().upper()
    # Oktav numarasını ayır
    octave = 4
    for i, ch in enumerate(note_str):
        if ch.isdigit():
            octave = int(ch)
            note_str = note_str[:i]
            break
    base_hz = _NOTE_MAP.get(note_str, 261.63)
    return base_hz * (2.0 ** (octave - 4))


def parse_melody_string(melody_text):
    """
    'C4 D4 E4 F4 G4' veya 'C,D#,E,A' biçimindeki metin girişini
    Hz frekans listesine çevirir.
    """
    tokens = melody_text.replace(',', ' ').split()
    return [_parse_note_string(t) for t in tokens if t]


def midi_to_frequencies(midi_path):
    """
    SRS FR1.22: MIDI dosyasından sıralı nota frekanslarını çıkarır.
    mido kütüphanesi kuruluysa kullanılır; değilse None döner.
    """
    try:
        import mido
        mid = mido.MidiFile(midi_path)
        notes = []
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    freq = 440.0 * (2.0 ** ((msg.note - 69) / 12.0))
                    notes.append(freq)
        return notes if notes else None
    except Exception:
        return None


def generate_singing_voice(y, sr, melody_notes=None,
                            midi_path=None, melody_text=None):
    """
    SRS FR1.21 – FR1.23: Konuşmadan Şarkı Sesi Üretimi.

    Giriş sesinin F0 konturunu verilen melodi notalarına hizalar ve
    PyWorld sentezleyici ile şarkıya dönüştürür.

    Parametreler
    ------------
    y             : kaynak ses dizisi
    sr            : örnekleme hızı
    melody_notes  : Hz cinsinden nota listesi (öncelik: en yüksek)
    midi_path     : MIDI dosya yolu (melody_notes yoksa kullanılır)
    melody_text   : 'C4 D4 E4' biçiminde metin (son alternatif)
    """
    # Melodi kaynağı seçimi: doğrudan liste > MIDI > metin > varsayılan gam
    if melody_notes is None:
        if midi_path is not None:
            melody_notes = midi_to_frequencies(midi_path)
        if melody_notes is None and melody_text:
            melody_notes = parse_melody_string(melody_text)
        if not melody_notes:
            melody_notes = _C_MAJOR_SCALE

    y = np.nan_to_num(y.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    _f0, t = pw.dio(y, sr)
    f0     = pw.stonemask(y, _f0, t, sr)
    sp     = pw.cheaptrick(y, f0, t, sr)
    ap     = pw.d4c(y, f0, t, sr)

    total_frames   = len(f0)
    n_notes        = len(melody_notes)
    frames_per_note = max(1, total_frames // n_notes)

    target_f0 = f0.copy()
    for i, note_hz in enumerate(melody_notes):
        start = i * frames_per_note
        end   = min(start + frames_per_note, total_frames)
        # Yalnızca sesli çerçevelerde melodiyi uygula
        voiced_mask = f0[start:end] > 0
        target_f0[start:end] = np.where(voiced_mask, note_hz, f0[start:end])

    synthesized = pw.synthesize(target_f0, sp, ap, sr)
    synthesized = np.nan_to_num(synthesized, nan=0.0, posinf=0.0, neginf=0.0)
    return synthesized
    """
    Spektral zarfı (formantları) 'env_ratio' oranında kaydırır.
    """
    if env_ratio == 1.0:
        return sp
    
    num_bins = sp.shape[1]
    freqs = np.linspace(0, sr // 2, num_bins)
    sp_shifted = np.zeros_like(sp)
    
    for i in range(sp.shape[0]):
        # Mevcut frekans dağılımı üzerinden interpolasyon fonksiyonu
        f_interp = interp1d(freqs, sp[i, :], kind='linear', bounds_error=False, fill_value=0.0)
        # Hedef frekanslar sıkıştırılır veya genleşir
        target_freqs = freqs / env_ratio
        sp_shifted[i, :] = f_interp(target_freqs)
        
    return sp_shifted

def change_gender_age(y, sr, f0_ratio=1.0, env_ratio=1.0):
    """
    SRS FR1.13 - FR1.16: Cinsiyet ve Yaş Dönüşümü. 
    PyWorld Vocoder kullanılarak Pitch (F0) ve Ses Yolu (Formant) değiştirilir.
    """
    y = y.astype(np.float64)
    # Pitch extraction using pyworld
    _f0, t = pw.dio(y, sr)
    f0 = pw.stonemask(y, _f0, t, sr)
    # Spectral envelope extraction
    sp = pw.cheaptrick(y, f0, t, sr)
    # Aperiodicity extraction
    ap = pw.d4c(y, f0, t, sr)
    
    # Modify f0 (Pitch Shifting)
    modified_f0 = f0 * f0_ratio
    
    # Modify Formants (Spectral Envelope Shifting)
    modified_sp = shift_formants(sp, sr, env_ratio)
    
    # Sentezleme
    synthesized = pw.synthesize(modified_f0, modified_sp, ap, sr)
    synthesized = np.nan_to_num(synthesized, nan=0.0, posinf=0.0, neginf=0.0)
    return synthesized

def change_emotion(y, sr, emotion="neutral", pitch_shift_st=None, speed_ratio=None, energy_scale=None):
    """
    SRS FR1.8 - FR1.12: Duygu Dönüşümü.
    Sesin perdesini (pitch), temposunu ve enerjisini manipüle eder.
    """
    y = np.nan_to_num(y.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    
    # Parametre Haritalaması: (f0_ratio, speed_ratio, energy_scale)
    emo_params = {
        "neutral": (1.0, 1.0, 1.0),
        "sad": (0.8, 0.8, 0.6),        # Lower pitch, slower, softer
        "angry": (1.3, 1.25, 1.4),     # Higher pitch, faster, louder
        "excited": (1.4, 1.2, 1.3),    # Very high pitch, faster, louder
        "calm": (0.9, 0.85, 0.7),      # Lower pitch, slightly slower, softer
        "whispered": (0.0, 0.9, 1.0),  # Flat unvoiced pitch
        "manual": (1.0, 1.0, 1.0)
    }
    
    b_f0, b_speed, b_energy = emo_params.get(emotion, (1.0, 1.0, 1.0))
    
    # GUI'den (app.py) gelen slider değerleri varsa onlara öncelik ver
    if pitch_shift_st is not None:
        final_f0_ratio = 2.0 ** (pitch_shift_st / 12.0)
    else:
        final_f0_ratio = b_f0
        
    if speed_ratio is not None:
        final_speed = speed_ratio
    else:
        final_speed = b_speed
        
    if energy_scale is not None:
        final_energy = energy_scale / 100.0 # Yüzdelik değerden çevirir
    else:
        final_energy = b_energy
    
    # Adım 1: Zaman esnetme (Time stretching) kullanarak logaritmik hız değiştirme
    if final_speed != 1.0:
        y = librosa.effects.time_stretch(y, rate=final_speed)
    
    # Adım 2: PyWorld kullanarak F0 Analizi ve Modifikasyonu
    _f0, t = pw.dio(y, sr)
    f0 = pw.stonemask(y, _f0, t, sr)
    sp = pw.cheaptrick(y, f0, t, sr)
    ap = pw.d4c(y, f0, t, sr)
    
    # Fısıltı için tüm sesi fısıtlıya dönüştür (f0 değerlerini sıfır yap, aperiodicity max)
    if emotion == "whispered":
        modified_f0 = np.zeros_like(f0)
    else:
        modified_f0 = f0 * final_f0_ratio
        
    synthesized = pw.synthesize(modified_f0, sp, ap, sr)
    
    # Adım 3: Enerji (Amplitude) değiştir
    synthesized = synthesized * final_energy
    
    return synthesized
