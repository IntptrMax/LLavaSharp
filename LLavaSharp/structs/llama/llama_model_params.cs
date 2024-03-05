using System;
using int32_t = System.Int32;

public struct llama_model_params
{
    public int32_t n_gpu_layers;
    public llama_split_mode split_mode;
    public int32_t main_gpu;

    public float[] tensor_split;
    public llama_progress_callback progress_callback;
    public IntPtr progress_callback_user_data;

    public llama_model_kv_override[] kv_overrides;
    public bool vocab_only;
    public bool use_mmap;
    public bool use_mlock;

    public static llama_model_params Default() =>
        new llama_model_params
        {
            n_gpu_layers = 0,
            main_gpu = 0,
            tensor_split = new float[] { 0 },
            progress_callback = null,
            progress_callback_user_data = IntPtr.Zero,
            kv_overrides = new llama_model_kv_override[] { },
            vocab_only = false,
            use_mmap = true,
            use_mlock = false,
        };
}

public delegate bool llama_progress_callback(float progress);

