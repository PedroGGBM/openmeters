// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maika Namuo
//
// 3D waterfall spectrogram renderer.
// Supports 4 rotation modes (0–3), matching the spectrogram's rotation convention.
//
// Rotation semantics for the waterfall:
//   0 (default) — frequency on x-axis (horizontal), oldest frames recede upward/back
//   1            — frequency on y-axis (vertical), oldest frames recede rightward/back
//   2            — frequency on x-axis, reversed (high freq at left)
//   3            — frequency on y-axis, reversed (high freq at bottom)

use crate::util::color::{rgba_with_alpha, sample_rgba_gradient};
use crate::visuals::render::common::{ClipTransform, SdfVertex, sdf_primitive};
use crate::visuals::spectrogram::render::SPECTROGRAM_PALETTE_SIZE;
use crate::visuals::spectrogram::processor::waterfall_mode::WaterfallFrame;
use iced::Rectangle;
use iced::advanced::graphics::Viewport;

// Vanishing-point offset from the near edge of the depth axis (0 = at edge, 1 = at far edge).
const VP_DEPTH_OFFSET: f32 = 0.04;
// Maximum ridge height as a fraction of the depth-axis extent at full perspective.
const MAX_AMP_HEIGHT: f32 = 0.52;
// Oldest frames fade out so the front stays vivid.
const DEPTH_FADE: f32 = 0.55;

#[derive(Debug, Clone)]
pub(crate) struct WaterfallParams {
    pub bounds: Rectangle,
    /// Oldest → newest.
    pub frames: Vec<WaterfallFrame>,
    pub num_bins: usize,
    /// Color gradient matching the spectrogram palette.
    pub palette: [[f32; 4]; SPECTROGRAM_PALETTE_SIZE],
    /// 0.0 = flat scroll, 1.0 = full 3D perspective.
    pub perspective: f32,
    /// 0 = freq-horizontal (default), 1 = freq-vertical,
    /// 2 = freq-horizontal-reversed, 3 = freq-vertical-reversed.
    pub rotation: u32,
    pub key: u64,
}

#[derive(Debug)]
pub(super) struct WaterfallPrimitive {
    params: WaterfallParams,
}

impl WaterfallPrimitive {
    pub fn new(params: WaterfallParams) -> Self {
        Self { params }
    }

    fn build_vertices(&self, viewport: &Viewport) -> Vec<SdfVertex> {
        let b = self.params.bounds;
        if b.width <= 0.0 || b.height <= 0.0 || self.params.frames.is_empty() {
            return Vec::new();
        }

        let clip = ClipTransform::from_viewport(viewport);
        let num_frames = self.params.frames.len();
        let num_bins = self.params.num_bins.max(1);
        let k = self.params.perspective.clamp(0.0, 1.0) * 5.0;

        let rotation = self.params.rotation & 3;
        // Bit 0: whether freq axis is vertical (1) or horizontal (0).
        // Bit 1: whether the freq axis direction is reversed.
        let freq_vertical = rotation & 1 == 1;
        let freq_flipped = rotation >= 2;

        let mut verts = Vec::with_capacity(num_frames * num_bins * 12);

        for (fi, frame) in self.params.frames.iter().enumerate() {
            if frame.len() < num_bins {
                continue;
            }

            // depth_t: 0 = newest (front/near edge), 1 = oldest (back, toward VP).
            let depth_t = 1.0 - fi as f32 / (num_frames - 1).max(1) as f32;
            let persp = 1.0 / (1.0 + depth_t * k);
            let alpha = 1.0 - depth_t * DEPTH_FADE;

            // Vanishing-point and floor positions differ by orientation.
            // "floor" is the zero-amplitude baseline; ridges grow away from it.
            let (vp_freq, vp_depth, near_depth) = if freq_vertical {
                // Freq on y, depth on x. VP is near the right edge.
                let vp_f = b.y + 0.5 * b.height;
                let vp_d = b.x + b.width * (1.0 - VP_DEPTH_OFFSET);
                let near = b.x; // newest frames at left
                (vp_f, vp_d, near)
            } else {
                // Freq on x, depth on y. VP is near the top.
                let vp_f = b.x + 0.5 * b.width;
                let vp_d = b.y + b.height * VP_DEPTH_OFFSET;
                let near = b.y + b.height; // newest frames at bottom
                (vp_f, vp_d, near)
            };

            // Floor screen-coordinate on the depth axis at this frame's depth.
            let floor_depth = if freq_vertical {
                // Oldest frames near VP (right), newest at left.
                vp_depth - (vp_depth - near_depth) * persp
            } else {
                // Oldest frames near VP (top), newest at bottom.
                vp_depth + (near_depth - vp_depth) * persp
            };

            // Maximum ridge extent on the depth axis at this frame's depth.
            let amp_h = if freq_vertical {
                b.width * MAX_AMP_HEIGHT * persp
            } else {
                b.height * MAX_AMP_HEIGHT * persp
            };

            // Screen positions of bin boundaries along the freq axis, perspective-compressed.
            let bin_fs: Vec<f32> = (0..=num_bins)
                .map(|bi| {
                    let t = bi as f32 / num_bins as f32;
                    let norm = if freq_flipped { 1.0 - t } else { t };
                    let raw_freq = if freq_vertical {
                        // Low freq at bottom (y = b.y + b.height) when not flipped.
                        b.y + (1.0 - norm) * b.height
                    } else {
                        b.x + norm * b.width
                    };
                    vp_freq + (raw_freq - vp_freq) * persp
                })
                .collect();

            // Amplitude at each bin boundary (average of adjacent bins).
            let amp_at: Vec<f32> = (0..=num_bins)
                .map(|bi| {
                    // When freq_flipped, frame[0] = high freq → boundary 0 reads high freq.
                    let raw = |i: usize| -> f32 {
                        let idx = if freq_flipped {
                            (num_bins - 1).saturating_sub(i)
                        } else {
                            i.min(num_bins - 1)
                        };
                        frame[idx]
                    };
                    if bi == 0 {
                        raw(0)
                    } else if bi == num_bins {
                        raw(num_bins - 1)
                    } else {
                        (raw(bi - 1) + raw(bi)) * 0.5
                    }
                })
                .collect();

            // Ridge screen-coordinate on the depth axis.
            let ridge_depth: Vec<f32> = amp_at
                .iter()
                .map(|&a| {
                    if freq_vertical {
                        // Ridge grows leftward from floor_depth (which is at some x).
                        // freq_vertical: floor on x-axis, ridge extends toward viewer (left).
                        floor_depth - amp_h * a
                    } else {
                        floor_depth - amp_h * a
                    }
                })
                .collect();

            // to_screen: (freq_coord, depth_coord) → (screen_x, screen_y)
            let to_screen = |fc: f32, dc: f32| -> (f32, f32) {
                if freq_vertical {
                    (dc, fc) // depth is x, freq is y
                } else {
                    (fc, dc) // freq is x, depth is y
                }
            };

            // --- filled trapezoids (floor to ridge) ---
            let floor_color = [0.0_f32, 0.0, 0.0, 0.0];
            for bi in 0..num_bins {
                let c_l = self.amp_color(amp_at[bi], alpha);
                let c_r = self.amp_color(amp_at[bi + 1], alpha);
                if c_l[3] < 0.005 && c_r[3] < 0.005 {
                    continue;
                }

                let (tl_x, tl_y) = to_screen(bin_fs[bi], ridge_depth[bi]);
                let (tr_x, tr_y) = to_screen(bin_fs[bi + 1], ridge_depth[bi + 1]);
                let (bl_x, bl_y) = to_screen(bin_fs[bi], floor_depth);
                let (br_x, br_y) = to_screen(bin_fs[bi + 1], floor_depth);

                let tl = clip.to_clip(tl_x, tl_y);
                let tr = clip.to_clip(tr_x, tr_y);
                let bl = clip.to_clip(bl_x, bl_y);
                let br = clip.to_clip(br_x, br_y);

                verts.extend_from_slice(&[
                    SdfVertex::solid(tl, c_l),
                    SdfVertex::solid(bl, floor_color),
                    SdfVertex::solid(br, floor_color),
                    SdfVertex::solid(tl, c_l),
                    SdfVertex::solid(br, floor_color),
                    SdfVertex::solid(tr, c_r),
                ]);
            }

            // --- glowing ridge line along the top of each frame ---
            for bi in 0..num_bins {
                if amp_at[bi] < 0.02 && amp_at[bi + 1] < 0.02 {
                    continue;
                }
                let ridge_alpha = alpha * 0.9;
                let c_l = self.ridge_color(amp_at[bi], ridge_alpha);
                let c_r = self.ridge_color(amp_at[bi + 1], ridge_alpha);
                let p0 = to_screen(bin_fs[bi], ridge_depth[bi]);
                let p1 = to_screen(bin_fs[bi + 1], ridge_depth[bi + 1]);
                verts.extend_from_slice(&thin_line_verts(p0, p1, c_l, c_r, 1.0, clip));
            }
        }

        verts
    }

    #[inline]
    fn amp_color(&self, amp: f32, alpha: f32) -> [f32; 4] {
        let c = sample_rgba_gradient(&self.params.palette, amp);
        rgba_with_alpha(c, c[3] * alpha)
    }

    #[inline]
    fn ridge_color(&self, amp: f32, alpha: f32) -> [f32; 4] {
        let c = sample_rgba_gradient(&self.params.palette, amp);
        [
            (c[0] * 1.35).min(1.0),
            (c[1] * 1.35).min(1.0),
            (c[2] * 1.35).min(1.0),
            c[3] * alpha,
        ]
    }
}

/// 1-pixel-wide antialiased line via a thin screen-space quad.
fn thin_line_verts(
    p0: (f32, f32),
    p1: (f32, f32),
    c0: [f32; 4],
    c1: [f32; 4],
    half_w: f32,
    clip: ClipTransform,
) -> [SdfVertex; 6] {
    let (dx, dy) = (p1.0 - p0.0, p1.1 - p0.1);
    let len = (dx * dx + dy * dy).sqrt().max(1e-5);
    let (nx, ny) = (-dy / len * half_w, dx / len * half_w);

    let tl = clip.to_clip(p0.0 + nx, p0.1 + ny);
    let bl = clip.to_clip(p0.0 - nx, p0.1 - ny);
    let tr = clip.to_clip(p1.0 + nx, p1.1 + ny);
    let br = clip.to_clip(p1.0 - nx, p1.1 - ny);

    [
        SdfVertex::solid(tl, c0),
        SdfVertex::solid(bl, c0),
        SdfVertex::solid(br, c1),
        SdfVertex::solid(tl, c0),
        SdfVertex::solid(br, c1),
        SdfVertex::solid(tr, c1),
    ]
}

sdf_primitive!(
    WaterfallPrimitive,
    Pipeline,
    u64,
    "Waterfall",
    TriangleList,
    |self| self.params.key
);
