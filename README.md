# spectral-prosody-rs

**Spectral analysis of prosodic features in agent communication.**

This crate provides a complete toolkit for extracting and analyzing prosodic features from signals: pitch contour tracking with semitone conversion, energy envelope analysis with dB scaling, MFCC-inspired spectral feature extraction with mel filter banks, and rhythmic pattern detection using autocorrelation. With 50 tests covering edge cases from silent signals to noisy multi-frequency inputs, it's designed to give agents an "ear" for the rhythms, stresses, and melodies of communication.

## Why This Matters

Prosody — the music of language — carries information that words alone cannot: emotion, intent, emphasis, uncertainty. For an AGI system, spectral prosody analysis is the bridge between raw audio and communicative understanding. Pitch contours reveal questions vs. statements. Energy envelopes encode emphasis and urgency. Rhythmic patterns distinguish languages and emotional states. MFCCs compress spectral shape into a fingerprint that machines can compare. This crate gives any Rust agent the ability to hear not just *what* is said, but *how* it's said — the foundation of empathetic communication.

## Quick Start

```toml
# Cargo.toml
[dependencies]
spectral-prosody-rs = "0.1.0"
```

```rust
use spectral_prosody_rs::pitch::PitchContour;
use spectral_prosody_rs::energy::{EnergyEnvelope, frame_energy};
use spectral_prosody_rs::rhythm::detect_rhythm;
use spectral_prosody_rs::spectral_features::{MelFilterBank, power_spectrum};

// Analyze a pitch contour (e.g., from a speech signal)
let pitch = PitchContour::new(
    vec![120.0, 130.0, 150.0, 145.0, 0.0, 140.0, 135.0],
    16000.0, // sample rate
    0.01,    // 10ms frame shift
);
println!("Mean pitch: {:.1} Hz", pitch.mean());
println!("Pitch std dev: {:.1} Hz", pitch.std_dev());
println!("Voicing ratio: {:.2}", pitch.voicing_ratio());
let semitones = pitch.to_semitones();

// Energy envelope analysis
let envelope = EnergyEnvelope::new(
    vec![0.1, 0.3, 0.8, 0.9, 0.7, 0.4, 0.2],
    16000.0, 0.01,
);
println!("Peak energy: {:.2}", envelope.peak());
println!("RMS: {:.4}", envelope.rms());
println!("Dynamic range: {:.1} dB", envelope.dynamic_range_db());

// Detect rhythmic patterns (tempo between 60-200 BPM)
let patterns = detect_rhythm(&envelope, 60.0, 200.0);
for p in &patterns {
    println!("Rhythm: {:.1} BPM (strength: {:.3})", p.bpm(0.01), p.strength);
}
```

## Architecture

| Module | Purpose |
|---|---|
| `pitch` | Pitch contour extraction, statistics, semitone conversion |
| `energy` | Energy envelope tracking, dB conversion, RMS, dynamic range |
| `rhythm` | Autocorrelation-based tempo detection, rhythmic pattern extraction |
| `spectral_features` | DCT, power spectrum, mel filter banks, MFCC computation |

## API Tour

### Pitch Analysis (`pitch`)

- **`PitchContour { values, sample_rate, frame_shift }`** — F0 trajectory
  - `.mean()` — Mean of voiced frames only
  - `.std_dev()` — Pitch variability
  - `.voicing_ratio()` — Fraction of voiced frames
  - `.to_semitones()` — Convert to relative semitones from mean
  - `.range()` — Min/max of voiced frames
  - `.duration()` — Total duration in seconds
  - `.jitter()` — Cycle-to-cycle pitch variation

### Energy Envelope (`energy`)

- **`EnergyEnvelope { values, sample_rate, frame_shift }`** — Energy over time
  - `.mean()`, `.peak()`, `.rms()` — Statistics
  - `.to_db(reference)` — Convert to decibels
  - `.dynamic_range_db()` — Peak-to-min in dB
- **`frame_energy(samples) → f64`** — Sum of squares for a single frame
- **`compute_envelope(signal, frame_len, frame_shift) → EnergyEnvelope`** — Frame-by-frame extraction

### Rhythm (`rhythm`)

- **`RhythmicPattern { period_frames, strength, phase }`** — Detected beat
  - `.bpm(frame_shift)` — Convert period to beats per minute
- **`autocorrelation(envelope, max_lag) → Vec<f64>`** — Normalized ACF
- **`detect_rhythm(envelope, min_bpm, max_bpm) → Vec<RhythmicPattern>`** — Peak-based tempo extraction

### Spectral Features (`spectral_features`)

- **`power_spectrum(signal) → Vec<f64>`** — DFT-based power spectrum
- **`dct(signal) → Vec<f64>`** — Discrete Cosine Transform (Type-II)
- **`MelFilterBank { num_filters, fft_size, sample_rate, low_freq, high_freq }`**
  - `.apply(spectrum) → Vec<f64>` — Apply filter bank
  - `::hz_to_mel(freq)`, `::mel_to_hz(mel)` — Frequency conversions
- **`mfcc(signal, num_filters, num_cepstral, sample_rate) → Vec<f64>`** — Full MFCC pipeline

## Performance

- Pitch/energy statistics are O(n) for n frames
- Power spectrum is O(n²) naive DFT — suitable for frame sizes ≤ 1024
- Autocorrelation is O(n × max_lag) — fast enough for real-time analysis
- Mel filter bank application is O(num_filters × fft_size)
- All operations are pure Rust with no external dependencies

## Ecosystem

Part of the **SuperInstance** family:

- [`topo-sonata-rs`](https://github.com/SuperInstance/topo-sonata-rs) — Topological analysis of musical structure
- [`optimal-transport-rs`](https://github.com/SuperInstance/optimal-transport-rs) — Compare spectral distributions
- [`witness-topology-rs`](https://github.com/SuperInstance/witness-topology-rs) — Topological features from audio point clouds
- [`sheaf-coherence-rs`](https://github.com/SuperInstance/sheaf-coherence-rs) — Multi-modal coherence
- [`renormalization-group-rs`](https://github.com/SuperInstance/renormalization-group-rs) — Multi-scale spectral analysis

## Ideas for Improvement

- **FFT acceleration** — Replace naive DFT with `rustfft` for real-time use
- **YIN/pyin pitch detection** — Add robust F0 estimation algorithms
- **Prosodic event detection** — Accents, boundaries, breaks
- **Spectral clustering** — Group similar prosodic patterns
- **Streaming API** — Frame-by-frame processing for live audio
- **Serialization** — Serde support for contour storage and replay

## License

MIT
