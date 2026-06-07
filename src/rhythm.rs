//! Rhythmic pattern analysis using spectral methods.

use crate::energy::EnergyEnvelope;

/// A rhythmic pattern detected in the signal.
#[derive(Debug, Clone)]
pub struct RhythmicPattern {
    pub period_frames: usize,
    pub strength: f64,
    pub phase: f64,
}

impl RhythmicPattern {
    pub fn new(period_frames: usize, strength: f64, phase: f64) -> Self {
        Self {
            period_frames,
            strength,
            phase,
        }
    }

    /// Beats per minute.
    pub fn bpm(&self, frame_shift: f64) -> f64 {
        if self.period_frames == 0 {
            return 0.0;
        }
        60.0 / (self.period_frames as f64 * frame_shift)
    }
}

/// Compute autocorrelation of an energy envelope.
pub fn autocorrelation(envelope: &EnergyEnvelope, max_lag: usize) -> Vec<f64> {
    let v = &envelope.values;
    if v.is_empty() {
        return vec![];
    }
    let n = v.len();
    let max_lag = max_lag.min(n / 2);

    let mean = v.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = v.iter().map(|&x| x - mean).collect();

    let norm0: f64 = centered.iter().map(|x| x * x).sum();

    let mut result = Vec::with_capacity(max_lag);
    for lag in 0..max_lag {
        let mut corr = 0.0;
        for i in 0..(n - lag) {
            corr += centered[i] * centered[i + lag];
        }
        if norm0 > 0.0 {
            result.push(corr / norm0);
        } else {
            result.push(0.0);
        }
    }
    result
}

/// Detect rhythmic patterns from autocorrelation peaks.
pub fn detect_rhythm(envelope: &EnergyEnvelope, min_bpm: f64, max_bpm: f64) -> Vec<RhythmicPattern> {
    if envelope.values.is_empty() {
        return vec![];
    }

    let frame_shift = envelope.frame_shift;
    let min_period = (60.0 / (max_bpm * frame_shift)) as usize;
    let max_period = (60.0 / (min_bpm * frame_shift)) as usize;

    let acf = autocorrelation(envelope, max_period + 1);

    let mut patterns = Vec::new();

    // Find peaks in autocorrelation
    for lag in min_period.max(1)..max_period.min(acf.len()) {
        if lag > 0 && lag < acf.len() - 1
            && acf[lag] > acf[lag - 1] && acf[lag] > acf[lag + 1] && acf[lag] > 0.1
        {
            patterns.push(RhythmicPattern::new(lag, acf[lag], 0.0));
        }
    }

    // Sort by strength (descending)
    patterns.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));

    patterns
}

/// Compute the tempo (dominant BPM) from energy envelope.
pub fn detect_tempo(envelope: &EnergyEnvelope) -> Option<f64> {
    let patterns = detect_rhythm(envelope, 30.0, 300.0);
    patterns.first().map(|p| p.bpm(envelope.frame_shift))
}

/// Compute inter-onset intervals from energy peaks.
pub fn inter_onset_intervals(envelope: &EnergyEnvelope, threshold: f64) -> Vec<f64> {
    let peaks = crate::energy::detect_peaks(envelope, threshold);
    if peaks.len() < 2 {
        return vec![];
    }
    peaks
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64 * envelope.frame_shift)
        .collect()
}

/// Compute rhythmic regularity (std dev of IOIs normalized by mean).
pub fn rhythmic_regularity(envelope: &EnergyEnvelope, threshold: f64) -> f64 {
    let iois = inter_onset_intervals(envelope, threshold);
    if iois.len() < 2 {
        return 0.0;
    }
    let mean = iois.iter().sum::<f64>() / iois.len() as f64;
    if mean < 1e-10 {
        return 0.0;
    }
    let variance = iois.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / iois.len() as f64;
    let std_dev = variance.sqrt();
    1.0 - (std_dev / mean).min(1.0) // 1.0 = perfectly regular, 0.0 = irregular
}

/// Compute spectral flux (frame-to-frame change in energy).
pub fn spectral_flux(envelope: &EnergyEnvelope) -> Vec<f64> {
    envelope
        .values
        .windows(2)
        .map(|w| (w[1] - w[0]).max(0.0))
        .collect()
}

/// Compute the rhythmic spectrum via DFT of the energy envelope.
pub fn rhythmic_spectrum(envelope: &EnergyEnvelope) -> Vec<f64> {
    let n = envelope.values.len();
    if n == 0 {
        return vec![];
    }

    let half = n / 2 + 1;
    let mut spectrum = Vec::with_capacity(half);

    let mean = envelope.values.iter().sum::<f64>() / n as f64;

    for k in 0..half {
        let mut re = 0.0;
        let mut im = 0.0;
        for i in 0..n {
            let angle = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
            re += (envelope.values[i] - mean) * angle.cos();
            im -= (envelope.values[i] - mean) * angle.sin();
        }
        spectrum.push(re * re + im * im);
    }
    spectrum
}

/// Find the dominant rhythm frequency from the rhythmic spectrum.
pub fn dominant_rhythm_frequency(envelope: &EnergyEnvelope) -> Option<f64> {
    let spec = rhythmic_spectrum(envelope);
    if spec.len() < 3 {
        return None;
    }

    // Skip DC (index 0) and look for peak
    let mut best_k = 1;
    let mut best_val = spec[1];
    for k in 2..spec.len() {
        if spec[k] > best_val {
            best_val = spec[k];
            best_k = k;
        }
    }

    if best_val < 1e-15 {
        return None;
    }

    // Convert bin to frequency
    let freq = best_k as f64 / (envelope.values.len() as f64 * envelope.frame_shift);
    Some(freq)
}

/// Compute n-tuple rhythmic features for classification.
pub fn rhythmic_feature_vector(envelope: &EnergyEnvelope, threshold: f64) -> Vec<f64> {
    let tempo = detect_tempo(envelope).unwrap_or(0.0);
    let regularity = rhythmic_regularity(envelope, threshold);
    let flux = spectral_flux(envelope);
    let mean_flux = if flux.is_empty() { 0.0 } else { flux.iter().sum::<f64>() / flux.len() as f64 };
    let peak = envelope.peak();
    let mean = envelope.mean();
    let dom_freq = dominant_rhythm_frequency(envelope).unwrap_or(0.0);

    vec![tempo, regularity, mean_flux, peak, mean, dom_freq]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_envelope(values: Vec<f64>) -> EnergyEnvelope {
        EnergyEnvelope::new(values, 16000.0, 0.01)
    }

    #[test]
    fn test_autocorrelation() {
        // Repeating pattern
        let env = make_envelope(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let acf = autocorrelation(&env, 4);
        assert_eq!(acf.len(), 4);
        assert!((acf[0] - 1.0).abs() < 1e-10); // lag 0 = 1.0
        assert!(acf[2] > acf[1]); // period-2 pattern
    }

    #[test]
    fn test_autocorrelation_empty() {
        let env = make_envelope(vec![]);
        assert!(autocorrelation(&env, 10).is_empty());
    }

    #[test]
    fn test_detect_rhythm() {
        // 5 Hz rhythm (period = 20 frames at 0.01s shift = 0.2s = 300 BPM)
        let period = 20;
        let mut values = Vec::new();
        for i in 0..200 {
            if i % period == 0 {
                values.push(1.0);
            } else {
                values.push(0.1);
            }
        }
        let env = make_envelope(values);
        let patterns = detect_rhythm(&env, 30.0, 600.0);
        // Should detect the period-20 pattern
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_detect_tempo() {
        let env = make_envelope(vec![1.0; 100]);
        let tempo = detect_tempo(&env);
        // Constant envelope might not have clear rhythm
        assert!(tempo.is_none() || tempo.unwrap() >= 0.0);
    }

    #[test]
    fn test_rhythmic_pattern_bpm() {
        let p = RhythmicPattern::new(10, 0.8, 0.0);
        // 60 / (10 * 0.01) = 600 BPM
        assert!((p.bpm(0.01) - 600.0).abs() < 1e-10);
    }

    #[test]
    fn test_inter_onset_intervals() {
        let env = make_envelope(vec![0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
        let iois = inter_onset_intervals(&env, 0.5);
        assert!(!iois.is_empty());
    }

    #[test]
    fn test_rhythmic_regularity_perfect() {
        // Perfect rhythm
        let mut values = vec![0.0; 100];
        for i in (0..100).step_by(10) {
            values[i] = 1.0;
        }
        let env = make_envelope(values);
        let reg = rhythmic_regularity(&env, 0.5);
        assert!(reg > 0.9);
    }

    #[test]
    fn test_rhythmic_regularity_irregular() {
        let env = make_envelope(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
        let reg = rhythmic_regularity(&env, 0.5);
        assert!(reg < 1.0);
    }

    #[test]
    fn test_spectral_flux() {
        let env = make_envelope(vec![1.0, 2.0, 1.0, 3.0]);
        let flux = spectral_flux(&env);
        assert_eq!(flux.len(), 3);
        assert!((flux[0] - 1.0).abs() < 1e-10);
        assert!((flux[1] - 0.0).abs() < 1e-10);
        assert!((flux[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rhythmic_spectrum() {
        let env = make_envelope(vec![1.0; 64]);
        let spec = rhythmic_spectrum(&env);
        assert!(spec.len() > 0);
    }

    #[test]
    fn test_dominant_rhythm_frequency() {
        let mut values = vec![0.0; 200];
        for i in (0..200).step_by(20) {
            values[i] = 1.0;
        }
        let env = make_envelope(values);
        let freq = dominant_rhythm_frequency(&env);
        assert!(freq.is_some());
    }

    #[test]
    fn test_rhythmic_feature_vector() {
        let env = make_envelope(vec![1.0, 0.5, 1.0, 0.5, 1.0, 0.5]);
        let fv = rhythmic_feature_vector(&env, 0.3);
        assert_eq!(fv.len(), 6);
        for v in &fv {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_pattern_zero_period() {
        let p = RhythmicPattern::new(0, 0.5, 0.0);
        assert_eq!(p.bpm(0.01), 0.0);
    }
}
