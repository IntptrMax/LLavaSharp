using System;
using System.Runtime.InteropServices;
using uint32_t = System.Int32;


[StructLayout(LayoutKind.Explicit, Pack = 8)]
public struct llama_context_params
{
    /// <summary>
    /// RNG seed, -1 for random
    /// </summary>
    [FieldOffset(0)] public uint32_t seed;

    /// <summary>
    /// text context, 0 = from model
    /// </summary>
    [FieldOffset(4)] public uint32_t n_ctx;

    /// <summary>
    /// logical maximum batch size that can be submitted to llama_decode
    /// </summary>
    [FieldOffset(8)] public uint32_t n_batch;

    /// <summary>
    /// physical maximum batch size
    /// </summary>
    [FieldOffset(12)] public uint32_t n_ubatch;

    /// <summary>
    /// max number of sequences (i.e. distinct states for recurrent models)
    /// </summary>
    [FieldOffset(16)] public uint32_t n_seq_max;

    /// <summary>
    /// number of threads to use for generation
    /// </summary>
    [FieldOffset(20)] public uint32_t n_threads;

    /// <summary>
    /// number of threads to use for batch processing
    /// </summary>
    [FieldOffset(24)] public uint32_t n_threads_batch;

    /// <summary>
    /// RoPE scaling type, from `enum llama_rope_scaling_type`
    /// </summary>
    [FieldOffset(28)] public llama_rope_scaling_type rope_scaling_type;

    /// <summary>
    /// whether to pool (sum) embedding results by sequence id(ignored if no pooling layer)
    /// </summary>
    [FieldOffset(32)] public llama_pooling_type pooling_type;

    // ref: https://github.com/ggerganov/llama.cpp/pull/2054
    /// <summary>
    /// RoPE base frequency, 0 = from model
    /// </summary>
    [FieldOffset(36)] public float rope_freq_base;

    /// <summary>
    /// RoPE frequency scaling factor, 0 = from model
    /// </summary>
    [FieldOffset(40)] public float rope_freq_scale;

    /// <summary>
    /// YaRN extrapolation mix factor, negative = from model
    /// </summary>
    [FieldOffset(44)] public float yarn_ext_factor;

    /// <summary>
    /// YaRN magnitude scaling factor
    /// </summary>
    [FieldOffset(48)] public float yarn_attn_factor;

    /// <summary>
    /// YaRN low correction dim
    /// </summary>
    [FieldOffset(52)] public float yarn_beta_fast;

    /// <summary>
    /// YaRN high correction dim
    /// </summary>
    [FieldOffset(56)] public float yarn_beta_slow;

    /// <summary>
    /// YaRN original context size
    /// </summary>
    [FieldOffset(60)] public uint32_t yarn_orig_ctx;

    /// <summary>
    /// defragment the KV cache if holes/size > thold, < 0 disabled (default)
    /// </summary>
    [FieldOffset(64)] public float defrag_thold;

    [FieldOffset(72)] public ggml_backend_sched_eval_callback cb_eval;
    [FieldOffset(80)] public IntPtr cb_eval_user_data;

    /// <summary>
    /// data type for K cache
    /// </summary>
    [FieldOffset(88)] public ggml_type type_k;

    /// <summary>
    /// data type for V cache
    /// </summary>
    [FieldOffset(92)] public ggml_type type_v;

    // Keep the booleans together to avoid misalignment during copy-by-value.

    /// <summary>
    /// the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
    /// </summary>
    [FieldOffset(96)] public bool logits_all;

    /// <summary>
    /// if true, extract embeddings (together with logits)
    /// </summary>
    [FieldOffset(97)] public bool embeddings;

    /// <summary>
    /// whether to offload the KQV ops (including the KV cache) to GPU
    /// </summary>
    [FieldOffset(98)] public bool offload_kqv; 

    // Abort callback
    // if it returns true, execution of llama_decode() will be aborted
    // currently works only with CPU execution
    [FieldOffset(104)] public ggml_abort_callback abort_callback;
    [FieldOffset(112)] public IntPtr abort_callback_data;
}

public delegate bool ggml_backend_sched_eval_callback(IntPtr t, bool ask, IntPtr user_data);
public delegate bool ggml_abort_callback(IntPtr user_data);