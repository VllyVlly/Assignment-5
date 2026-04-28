#include "image_proc.h"
#include <cmath>
#include <algorithm>
#include <random>

void initialize_image_proc(image_proc_args *args, size_t w, size_t h, uint64_t seed) {
    args->width = w;
    args->height = h;
    size_t n = w * h;
    
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    args->r_channel.assign(n, 0.0f);
    args->g_channel.assign(n, 0.0f);
    args->b_channel.assign(n, 0.0f);
    args->output.assign(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        args->r_channel[i] = dist(gen);
        args->g_channel[i] = dist(gen);
        args->b_channel[i] = dist(gen);
    }
}

// -------------------------------------------------------------------------
// Functions
// You should not modify these functions
// -------------------------------------------------------------------------

// Image Process A
__attribute__((noinline)) 
float apply_gain(float v) { return v * 1.05f; }

__attribute__((noinline)) 
float apply_shift(float v) { return v + 0.02f; }

__attribute__((noinline)) 
float apply_limit(float v) { return (v > 1.0f) ? 1.0f : v; }

__attribute__((noinline)) 
float color_correct(float v) {
    return apply_limit(apply_shift(apply_gain(v)));
}

// Image Process B
__attribute__((noinline)) 
float compute_gray(float r, float g, float b) {
    return (r * 0.299f) + (g * 0.587f) + (b * 0.114f);
}

__attribute__((noinline)) 
float enhance_contrast(float gray){
    // imadjust logic: mapping 0.05-0.95 to 0.0-1.0
    float adjusted = std::clamp((gray - 0.05f) / 0.90f, 0.0f, 1.0f);
    
    // Sigmoidal S-Curve for "punchy" contrast
    return adjusted * adjusted * (3.0f - 2.0f * adjusted);
}

__attribute__((noinline)) 
float calculate_gain(float intensity) {
    float g1 = intensity * 0.5f;
    float g2 = g1 * g1 + 0.1f;
    float g3 = std::sqrt(g2);
    return (g3 > 1.0f) ? (1.0f / g3) : (g3 * 0.95f);
}

__attribute__((noinline)) 
float hdr_compress(float val) {
    float gain = calculate_gain(val * 1.2f);
    float result = val * gain;
    return result / (1.0f + result);
}

__attribute__((noinline)) 
float complex_mask_logic(float gray, float r, float g, float b, float thresh) {
    const float p0=0.11f, p1=0.22f, p2=0.33f, p3=0.44f, p4=0.55f;
    const float p5=0.66f, p6=0.77f, p7=0.88f, p8=0.99f, p9=1.01f;
    
    float mask = 0.0f;
    if (gray > thresh) {
        mask = (r * p0) + (g * p1) - (b * p2) + p9;
        if (mask > 0.8f) mask *= p3;
        else mask += p4;
    } else {
        mask = (r * p5) - (g * p6) + (b * p7) - p8;
        if (mask < 0.2f) mask += p1;
        else mask *= p2;
    }

    float noise = std::sin(gray * p0) * std::cos(r * p1);
    float final_val = (mask * 0.7f) + (noise * 0.3f);
    
    return std::clamp(final_val, 0.0f, 1.0f);
}

__attribute__((noinline)) 
float importance_weight(float val) {
    static const float lut[] = {0.0f, 0.3f, 1.0f, 0.3f, 0.0f};
    float scaled = val * 4.0f;
    int idx = std::clamp(static_cast<int>(scaled), 0, 4);
    float weight = scaled - static_cast<float>(idx);
    
    if (idx < 4) return lut[idx] * (1.0f - weight) + lut[idx + 1] * weight;
    return lut[4];
}

__attribute__((noinline)) 
float fast_activate(float val) {
    return val / (1.0f + std::abs(val));
}
// -------------------------------------------------------------------------
// End of Functions
// -------------------------------------------------------------------------


// -------------------------------------------------------------------------
// Naive Implementation: Image Processing
// -------------------------------------------------------------------------
void naive_image_proc(image_proc_args& args) {
    const size_t w = args.width;
    const size_t h = args.height;
    float* __restrict__ out = args.output.data();
    const float* __restrict__ r = args.r_channel.data();
    const float* __restrict__ g = args.g_channel.data();
    const float* __restrict__ b = args.b_channel.data();
    const float threshold = args.threshold;

    for (size_t y = 0; y < h; ++y)  {
        for (size_t x = 0; x < w; ++x){
            size_t i = y * w + x;

            // Stage 1: Update RGB Value
            float r_val = color_correct(r[i]);
            float g_val = color_correct(g[i]);
            float b_val = color_correct(b[i]);

            // Stage 2: Luminance Extraction
            float gray = compute_gray(r_val, g_val, b_val);

            // Stage 3: Contrast Enhancement
            float grayEnhance = enhance_contrast(gray);

            // Stage 4: HDR Compression
            float compress_val = hdr_compress(grayEnhance);

            // Stage 5: Masking
            float mask = complex_mask_logic(compress_val, r_val, g_val, b_val, threshold);

            // Stage 6: Importance Weighting
            float weight = importance_weight(mask);

            // Output
            out[i] = std::clamp(compress_val * weight, 0.0f, 1.0f);

        }
    }
}

// -------------------------------------------------------------------------
// TODO: Student Implementation
// -------------------------------------------------------------------------
void stu_image_proc(image_proc_args& args) {
    const size_t w = args.width;
    const size_t h = args.height;
    float* __restrict__ out = args.output.data();
    const float* __restrict__ r = args.r_channel.data();
    const float* __restrict__ g = args.g_channel.data();
    const float* __restrict__ b = args.b_channel.data();
    const float threshold = args.threshold;

    constexpr float gain_scale = 1.05f;
    constexpr float shift = 0.02f;
    constexpr float gray_r = 0.299f;
    constexpr float gray_g = 0.587f;
    constexpr float gray_b = 0.114f;
    constexpr float contrast_low = 0.05f;
    constexpr float contrast_scale = 1.0f / 0.90f;
    constexpr float hdr_input_scale = 1.2f;

    constexpr float p0 = 0.11f;
    constexpr float p1 = 0.22f;
    constexpr float p2 = 0.33f;
    constexpr float p3 = 0.44f;
    constexpr float p4 = 0.55f;
    constexpr float p5 = 0.66f;
    constexpr float p6 = 0.77f;
    constexpr float p7 = 0.88f;
    constexpr float p8 = 0.99f;
    constexpr float p9 = 1.01f;

    constexpr float lut0 = 0.0f;
    constexpr float lut1 = 0.3f;
    constexpr float lut2 = 1.0f;
    constexpr float lut3 = 0.3f;
    constexpr float lut4 = 0.0f;

    for (size_t y = 0; y < h; ++y) {
        const size_t row = y * w;
        for (size_t x = 0; x < w; ++x) {
            const size_t i = row + x;

            // Stage 1: color_correct(apply_gain/apply_shift/apply_limit)
            float r_val = r[i] * gain_scale + shift;
            float g_val = g[i] * gain_scale + shift;
            float b_val = b[i] * gain_scale + shift;
            r_val = (r_val > 1.0f) ? 1.0f : r_val;
            g_val = (g_val > 1.0f) ? 1.0f : g_val;
            b_val = (b_val > 1.0f) ? 1.0f : b_val;

            // Stage 2: compute_gray
            const float gray = r_val * gray_r + g_val * gray_g + b_val * gray_b;

            // Stage 3: enhance_contrast
            const float adjusted =
                std::clamp((gray - contrast_low) * contrast_scale, 0.0f, 1.0f);
            const float gray_enhance =
                adjusted * adjusted * (3.0f - 2.0f * adjusted);

            // Stage 4: hdr_compress(calculate_gain)
            const float intensity = gray_enhance * hdr_input_scale;
            const float g1 = intensity * 0.5f;
            const float g2 = g1 * g1 + 0.1f;
            const float g3 = std::sqrt(g2);
            const float hdr_gain = (g3 > 1.0f) ? (1.0f / g3) : (g3 * 0.95f);
            const float hdr_product = gray_enhance * hdr_gain;
            const float compress_val = hdr_product / (1.0f + hdr_product);

            // Stage 5: complex_mask_logic
            float mask;
            if (compress_val > threshold) {
                mask = r_val * p0 + g_val * p1 - b_val * p2 + p9;
                mask = (mask > 0.8f) ? (mask * p3) : (mask + p4);
            } else {
                mask = r_val * p5 - g_val * p6 + b_val * p7 - p8;
                mask = (mask < 0.2f) ? (mask + p1) : (mask * p2);
            }

            const float noise = std::sin(compress_val * p0) * std::cos(r_val * p1);
            const float final_mask =
                std::clamp(mask * 0.7f + noise * 0.3f, 0.0f, 1.0f);

            // Stage 6: importance_weight
            const float scaled = final_mask * 4.0f;
            const int idx = std::clamp(static_cast<int>(scaled), 0, 4);
            const float weight_frac = scaled - static_cast<float>(idx);

            float weight;
            switch (idx) {
                case 0:
                    weight = lut0 * (1.0f - weight_frac) + lut1 * weight_frac;
                    break;
                case 1:
                    weight = lut1 * (1.0f - weight_frac) + lut2 * weight_frac;
                    break;
                case 2:
                    weight = lut2 * (1.0f - weight_frac) + lut3 * weight_frac;
                    break;
                case 3:
                    weight = lut3 * (1.0f - weight_frac) + lut4 * weight_frac;
                    break;
                default:
                    weight = lut4;
                    break;
            }

            out[i] = std::clamp(compress_val * weight, 0.0f, 1.0f);
        }
    }
}


// -------------------------------------------------------------------------
// Wrappers and Utilities
// -------------------------------------------------------------------------
void naive_image_proc_wrapper(void *ctx) {
    auto &args = *static_cast<image_proc_args *>(ctx);
    naive_image_proc(args);
}

void stu_image_proc_wrapper(void *ctx) {
    auto &args = *static_cast<image_proc_args *>(ctx);
    stu_image_proc(args);
}

bool image_proc_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);
    auto &stu = *static_cast<image_proc_args *>(stu_ctx);
    auto &ref = *static_cast<image_proc_args *>(ref_ctx);

    if (stu.output.size() != ref.output.size()) {
        debug_log("DEBUG: image_proc size mismatch: stu={} ref={}\n",
                  stu.output.size(),
                  ref.output.size());
        return false;
    }

    for (size_t i = 0; i < ref.output.size(); ++i) {
        const double err =
            std::abs(static_cast<double>(stu.output[i]) -
                     static_cast<double>(ref.output[i]));
        if (err > 1e-4) {
            debug_log("DEBUG: image_proc fail at {}: ref={} stu={} err={}\n",
                      i,
                      ref.output[i],
                      stu.output[i],
                      err);
            return false;
        }
    }
    debug_log("DEBUG: image_proc_check passed. size={}\n", ref.output.size());
    return true;
}
