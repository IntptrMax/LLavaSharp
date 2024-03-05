using uint32_t = System.Int32;
using int8_t = System.SByte;
using System;

public struct llama_context_params
{
    public uint32_t seed;              // RNG seed, -1 for random
    public uint32_t n_ctx;             // text context, 0 = from model
    public uint32_t n_batch;           // prompt processing maximum batch size
    public uint32_t n_threads;         // number of threads to use for generation
    public uint32_t n_threads_batch;   // number of threads to use for batch processing
    public int8_t rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`

    // ref: https://github.com/ggerganov/llama.cpp/pull/2054
    public float rope_freq_base;   // RoPE base frequency, 0 = from model
    public float rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
    public float yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
    public float yarn_attn_factor; // YaRN magnitude scaling factor
    public float yarn_beta_fast;   // YaRN low correction dim
    public float yarn_beta_slow;   // YaRN high correction dim
    public uint32_t yarn_orig_ctx;    // YaRN original context size

    public GgmlBackendSchedEvalCallback cb_eval;
    public IntPtr cb_eval_user_data;

    public ggml_type type_k; // data type for K cache
    public ggml_type type_v; // data type for V cache

    // Keep the booleans together to avoid misalignment during copy-by-value.
    public bool mul_mat_q;   // if true, use experimental mul_mat_q kernels (DEPRECATED - always true)
    public bool logits_all;  // the llama_eval() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
    public bool embedding;   // embedding mode only
    public bool offload_kqv; // whether to offload the KQV ops (including the KV cache) to GPU
}

public delegate bool GgmlBackendSchedEvalCallback(ref IntPtr t, bool ask, IntPtr user_data);
