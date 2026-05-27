// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maika Namuo
//
// Self-contained waterfall DSP sub-processor.
// Used by SpectrogramProcessor when display_mode == Waterfall.

use crate::util::audio::{
    DB_FLOOR, WindowKind, mixdown_into_deque, power_to_db, window_coefficients,
};
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex32;
use std::collections::VecDeque;
use std::sync::Arc;

/// Number of log-spaced display bins in each waterfall frame.
pub(crate) const WATERFALL_NUM_BINS: usize = 256;

/// Frequency range shown in the waterfall display.
pub(crate) const WATERFALL_FREQ_MIN: f32 = 20.0;
pub(crate) const WATERFALL_FREQ_MAX: f32 = 20_000.0;

/// A single normalized waterfall frame: `WATERFALL_NUM_BINS` values in [0, 1].
/// `Arc` keeps snapshot cloning O(1).
pub type WaterfallFrame = Arc<[f32]>;

// (fft_bin_lo..fft_bin_hi) for each display bin — precomputed once.
type BinMap = Box<[(usize, usize)]>;

pub(super) struct WaterfallSubProc {
    fft: Arc<dyn RealToComplex<f32>>,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex32>,
    scratch: Vec<Complex32>,
    window: Arc<[f32]>,
    mono_buf: VecDeque<f32>,
    bin_map: BinMap,
    tilt_weights: Box<[f32]>,

    pub floor_db: f32,
    pub tilt_db_per_octave: f32,

    fft_size: usize,
    hop_size: usize,
    sample_rate: f32,
}

impl WaterfallSubProc {
    pub fn new(fft_size: usize, hop_size: usize, sample_rate: f32) -> Self {
        let fft_size = fft_size.next_power_of_two().max(64);
        let mut planner = RealFftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let window = window_coefficients(WindowKind::BlackmanHarris, fft_size);
        let bin_map = Self::make_bin_map(fft_size, sample_rate);
        let tilt_weights = vec![1.0_f32; WATERFALL_NUM_BINS].into_boxed_slice();
        let fft_input = fft.make_input_vec();
        let fft_output = fft.make_output_vec();
        let scratch = fft.make_scratch_vec();
        Self {
            fft,
            fft_input,
            fft_output,
            scratch,
            window,
            mono_buf: VecDeque::new(),
            bin_map,
            tilt_weights,
            floor_db: -80.0,
            tilt_db_per_octave: 0.0,
            fft_size,
            hop_size,
            sample_rate,
        }
    }

    pub fn rebuild_if_needed(&mut self, fft_size: usize, hop_size: usize, sample_rate: f32) {
        let fft_size = fft_size.next_power_of_two().max(64);
        if fft_size != self.fft_size
            || (sample_rate - self.sample_rate).abs() > f32::EPSILON
        {
            let mut planner = RealFftPlanner::new();
            self.fft = planner.plan_fft_forward(fft_size);
            self.window = window_coefficients(WindowKind::BlackmanHarris, fft_size);
            self.bin_map = Self::make_bin_map(fft_size, sample_rate);
            self.tilt_weights = Self::make_tilt_weights(
                &self.bin_map, fft_size, sample_rate, self.tilt_db_per_octave,
            );
            self.fft_input = self.fft.make_input_vec();
            self.fft_output = self.fft.make_output_vec();
            self.scratch = self.fft.make_scratch_vec();
            self.mono_buf.clear();
            self.fft_size = fft_size;
            self.sample_rate = sample_rate;
        }
        self.hop_size = hop_size;
    }

    pub fn update_tilt(&mut self, tilt_db_per_octave: f32) {
        if (tilt_db_per_octave - self.tilt_db_per_octave).abs() > 1e-3 {
            self.tilt_db_per_octave = tilt_db_per_octave;
            self.tilt_weights = Self::make_tilt_weights(
                &self.bin_map,
                self.fft_size,
                self.sample_rate,
                tilt_db_per_octave,
            );
        }
    }

    /// Append audio samples (any number of channels) to the internal queue.
    pub fn feed(&mut self, samples: &[f32], channels: usize) {
        mixdown_into_deque(&mut self.mono_buf, samples, channels);
    }

    /// Process all complete windows from the internal queue.
    /// Returns one `WaterfallFrame` per complete hop.
    pub fn drain_frames(&mut self) -> Vec<WaterfallFrame> {
        let mut frames = Vec::new();
        let hop = self.hop_size.clamp(1, self.fft_size);

        while self.mono_buf.len() >= self.fft_size {
            let window = self.window.clone();
            for (dst, (&src, &w)) in self
                .fft_input
                .iter_mut()
                .zip(self.mono_buf.iter().take(self.fft_size).zip(window.iter()))
            {
                *dst = src * w;
            }
            if self
                .fft
                .process_with_scratch(
                    &mut self.fft_input,
                    &mut self.fft_output,
                    &mut self.scratch,
                )
                .is_ok()
            {
                frames.push(self.make_frame());
            }
            self.mono_buf.drain(..hop);
        }
        frames
    }

    fn make_frame(&self) -> WaterfallFrame {
        let scale = 1.0 / (self.fft_size as f32 * self.fft_size as f32);
        let floor = self.floor_db;
        let db_range = (0.0_f32 - floor).max(1.0);

        let frame: Vec<f32> = self
            .bin_map
            .iter()
            .zip(self.tilt_weights.iter())
            .map(|(&(lo, hi), &w)| {
                let power = self.fft_output[lo..hi]
                    .iter()
                    .map(|c| c.norm_sqr())
                    .fold(0.0_f32, f32::max)
                    * scale;
                let db = power_to_db(power * w * w, DB_FLOOR);
                ((db - floor) / db_range).clamp(0.0, 1.0)
            })
            .collect();

        Arc::from(frame.as_slice())
    }

    fn make_bin_map(fft_size: usize, sample_rate: f32) -> BinMap {
        let spectrum_len = fft_size / 2 + 1;
        let bin_hz = sample_rate / fft_size as f32;
        let nyquist = sample_rate * 0.5;
        let freq_max = WATERFALL_FREQ_MAX.min(nyquist);
        let freq_min = WATERFALL_FREQ_MIN.min(freq_max * 0.5);
        let log_min = freq_min.max(1.0).ln();
        let log_range = (freq_max / freq_min.max(1.0)).ln().max(0.01);

        (0..WATERFALL_NUM_BINS)
            .map(|i| {
                let t0 = i as f32 / WATERFALL_NUM_BINS as f32;
                let t1 = (i + 1) as f32 / WATERFALL_NUM_BINS as f32;
                let f_lo = (log_min + t0 * log_range).exp();
                let f_hi = (log_min + t1 * log_range).exp();
                let b_lo = ((f_lo / bin_hz).floor() as usize).clamp(1, spectrum_len - 1);
                let b_hi = ((f_hi / bin_hz).ceil() as usize + 1).clamp(b_lo + 1, spectrum_len);
                (b_lo, b_hi)
            })
            .collect()
    }

    fn make_tilt_weights(
        bin_map: &[(usize, usize)],
        fft_size: usize,
        sample_rate: f32,
        tilt_db_per_oct: f32,
    ) -> Box<[f32]> {
        if tilt_db_per_oct.abs() < 1e-3 {
            return vec![1.0; bin_map.len()].into_boxed_slice();
        }
        let bin_hz = sample_rate / fft_size as f32;
        let ref_hz: f32 = 1000.0;
        bin_map
            .iter()
            .map(|&(lo, hi)| {
                let center_hz = (lo + hi) as f32 * 0.5 * bin_hz;
                let octaves = (center_hz / ref_hz).max(1e-6).log2();
                10.0_f32.powf(tilt_db_per_oct * octaves / 20.0)
            })
            .collect()
    }
}
