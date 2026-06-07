//! MFCC-inspired feature extraction for prosodic analysis.

/// Compute the Discrete Cosine Transform (Type-II) of a signal.
pub fn dct(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }
    let mut result = Vec::with_capacity(n);
    for k in 0..n {
        let mut sum = 0.0;
        for i in 0..n {
            sum += signal[i] * (std::f64::consts::PI * (i as f64 + 0.5) * k as f64 / n as f64).cos();
        }
        result.push(sum);
    }
    result
}

/// Compute the power spectrum using DFT.
pub fn power_spectrum(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }
    let nfft = n;
    let half = nfft / 2 + 1;
    let mut spectrum = Vec::with_capacity(half);

    for k in 0..half {
        let mut re = 0.0;
        let mut im = 0.0;
        for i in 0..nfft {
            let angle = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / nfft as f64;
            re += signal[i] * angle.cos();
            im -= signal[i] * angle.sin();
        }
        spectrum.push((re * re + im * im) / nfft as f64);
    }

    spectrum
}

/// Mel-scale filter bank.
pub struct MelFilterBank {
    pub num_filters: usize,
    pub fft_size: usize,
    pub sample_rate: f64,
    pub low_freq: f64,
    pub high_freq: f64,
}

impl MelFilterBank {
    pub fn new(num_filters: usize, fft_size: usize, sample_rate: f64, low_freq: f64, high_freq: f64) -> Self {
        Self {
            num_filters,
            fft_size,
            sample_rate,
            low_freq,
            high_freq,
        }
    }

    /// Convert frequency to mel scale.
    pub fn hz_to_mel(freq: f64) -> f64 {
        2595.0 * (1.0 + freq / 700.0).log10()
    }

    /// Convert mel scale to frequency.
    pub fn mel_to_hz(mel: f64) -> f64 {
        700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
    }

    /// Apply filter bank to power spectrum.
    pub fn apply(&self, spectrum: &[f64]) -> Vec<f64> {
        let num_bins = spectrum.len();
        let low_mel = Self::hz_to_mel(self.low_freq);
        let high_mel = Self::hz_to_mel(self.high_freq);

        let mel_points: Vec<f64> = (0..=self.num_filters + 1)
            .map(|i| low_mel + (high_mel - low_mel) * i as f64 / (self.num_filters + 1) as f64)
            .collect();

        let bin_points: Vec<usize> = mel_points
            .iter()
            .map(|&m| {
                let hz = Self::mel_to_hz(m);
                ((num_bins as f64) * hz / self.sample_rate).floor() as usize
            })
            .collect();

        let mut filter_energies = Vec::with_capacity(self.num_filters);
        for i in 0..self.num_filters {
            let left = bin_points[i];
            let center = bin_points[i + 1];
            let right = bin_points[i + 2];

            let mut energy = 0.0_f64;
            for k in left..center.min(num_bins) {
                if center > left {
                    let weight = (k - left) as f64 / (center - left) as f64;
                    energy += spectrum[k] * weight;
                }
            }
            for k in center..right.min(num_bins) {
                if right > center {
                    let weight = (right - k) as f64 / (right - center) as f64;
                    energy += spectrum[k] * weight;
                }
            }
            filter_energies.push(energy.max(1e-10));
        }

        filter_energies
    }
}

/// Compute MFCCs from a signal frame.
pub fn compute_mfcc(
    signal: &[f64],
    num_cepstral: usize,
    num_filters: usize,
    sample_rate: f64,
) -> Vec<f64> {
    if signal.is_empty() {
        return vec![];
    }

    // Apply Hamming window
    let n = signal.len();
    let windowed: Vec<f64> = signal
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let w = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
            s * w
        })
        .collect();

    // Power spectrum
    let spec = power_spectrum(&windowed);

    // Mel filter bank
    let fbank = MelFilterBank::new(num_filters, n, sample_rate, 0.0, sample_rate / 2.0);
    let mel_energies = fbank.apply(&spec);

    // Log mel energies
    let log_mel: Vec<f64> = mel_energies.iter().map(|&e| e.ln()).collect();

    // DCT
    let cepstral = dct(&log_mel);

    // Take first num_cepstral coefficients
    cepstral.into_iter().take(num_cepstral).collect()
}

/// Compute delta (first derivative) features.
pub fn delta(features: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    if features.len() < 2 * n + 1 {
        return vec![vec![0.0; features.first().map_or(0, |f| f.len())]];
    }

    let dim = features[0].len();
    let mut deltas = Vec::with_capacity(features.len());

    for t in 0..features.len() {
        let mut d = vec![0.0; dim];
        let mut denom = 0.0;
        for i in 1..=n {
            let t_prev = t.saturating_sub(i);
            let t_next = (t + i).min(features.len() - 1);
            for j in 0..dim {
                d[j] += i as f64 * (features[t_next][j] - features[t_prev][j]);
            }
            denom += 2.0 * (i as f64) * (i as f64);
        }
        if denom > 0.0 {
            for j in 0..dim {
                d[j] /= denom;
            }
        }
        deltas.push(d);
    }

    deltas
}

/// Feature vector combining MFCC, delta, and delta-delta.
pub struct ProsodicFeatures {
    pub mfcc: Vec<Vec<f64>>,
    pub delta: Vec<Vec<f64>>,
    pub delta_delta: Vec<Vec<f64>>,
}

impl ProsodicFeatures {
    /// Extract full prosodic feature set from signal.
    pub fn extract(
        signal: &[f64],
        sample_rate: f64,
        frame_length: usize,
        frame_shift: usize,
        num_cepstral: usize,
        num_filters: usize,
    ) -> Self {
        let mut mfcc_frames = Vec::new();
        let mut start = 0;
        while start + frame_length <= signal.len() {
            let frame = &signal[start..start + frame_length];
            let coeffs = compute_mfcc(frame, num_cepstral, num_filters, sample_rate);
            mfcc_frames.push(coeffs);
            start += frame_shift;
        }

        let d = delta(&mfcc_frames, 2);
        let dd = delta(&d, 2);

        Self {
            mfcc: mfcc_frames,
            delta: d,
            delta_delta: dd,
        }
    }

    /// Total feature dimension per frame.
    pub fn feature_dim(&self) -> usize {
        let mfcc_dim = self.mfcc.first().map_or(0, |f| f.len());
        let d_dim = self.delta.first().map_or(0, |f| f.len());
        let dd_dim = self.delta_delta.first().map_or(0, |f| f.len());
        mfcc_dim + d_dim + dd_dim
    }

    /// Number of frames.
    pub fn num_frames(&self) -> usize {
        self.mfcc.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_dc() {
        let signal = vec![1.0; 4];
        let result = dct(&signal);
        // DC component should be sum
        assert!(result[0] > 0.0);
    }

    #[test]
    fn test_dct_empty() {
        assert!(dct(&[]).is_empty());
    }

    #[test]
    fn test_power_spectrum_sine() {
        let signal: Vec<f64> = (0..256)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / 256.0).sin())
            .collect();
        let spec = power_spectrum(&signal);
        assert_eq!(spec.len(), 129);
        // Should have a peak near bin 10
        let max_bin = spec.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert!((max_bin as i32 - 10).abs() <= 2);
    }

    #[test]
    fn test_power_spectrum_empty() {
        assert!(power_spectrum(&[]).is_empty());
    }

    #[test]
    fn test_mel_conversion() {
        let hz = 1000.0;
        let mel = MelFilterBank::hz_to_mel(hz);
        let back = MelFilterBank::mel_to_hz(mel);
        assert!((back - hz).abs() < 0.01);
    }

    #[test]
    fn test_mel_filterbank_apply() {
        let spec = vec![1.0; 129];
        let fbank = MelFilterBank::new(26, 256, 16000.0, 0.0, 8000.0);
        let energies = fbank.apply(&spec);
        assert_eq!(energies.len(), 26);
        // All energies should be positive
        for e in &energies {
            assert!(*e > 0.0);
        }
    }

    #[test]
    fn test_compute_mfcc() {
        let signal: Vec<f64> = (0..400)
            .map(|i| (2.0 * std::f64::consts::PI * 200.0 * i as f64 / 16000.0).sin())
            .collect();
        let mfcc = compute_mfcc(&signal, 13, 26, 16000.0);
        assert_eq!(mfcc.len(), 13);
    }

    #[test]
    fn test_compute_mfcc_empty() {
        let mfcc = compute_mfcc(&[], 13, 26, 16000.0);
        assert!(mfcc.is_empty());
    }

    #[test]
    fn test_delta() {
        let features = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
            vec![5.0, 6.0],
        ];
        let d = delta(&features, 1);
        assert_eq!(d.len(), 5);
        // Deltas should be approximately constant for linear sequence
        assert!(d[2][0] > 0.0);
    }

    #[test]
    fn test_delta_short() {
        let features = vec![vec![1.0]];
        let d = delta(&features, 2);
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn test_prosodic_features_extract() {
        let signal: Vec<f64> = (0..4800)
            .map(|i| (2.0 * std::f64::consts::PI * 200.0 * i as f64 / 16000.0).sin())
            .collect();
        let features = ProsodicFeatures::extract(&signal, 16000.0, 400, 200, 13, 26);
        assert!(features.num_frames() > 0);
        assert_eq!(features.feature_dim(), 39); // 13 * 3
    }

    #[test]
    fn test_mfcc_values_reasonable() {
        let signal: Vec<f64> = (0..800)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 16000.0).sin())
            .collect();
        let mfcc = compute_mfcc(&signal, 13, 26, 16000.0);
        // C0 should be non-zero for a sine wave
        assert!(mfcc[0].abs() > 0.0);
        // Values should be finite
        for c in &mfcc {
            assert!(c.is_finite());
        }
    }
}
