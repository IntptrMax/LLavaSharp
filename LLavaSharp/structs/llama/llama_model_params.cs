using System;
using System.Runtime.InteropServices;
using int32_t = System.Int32;

[StructLayout(LayoutKind.Explicit, Pack = 8)]
public struct llama_model_params
{
    [FieldOffset(0x0000)] public int32_t n_gpu_layers;
    [FieldOffset(0x0004)] public llama_split_mode split_mode;
    [FieldOffset(0x0008)] public int32_t main_gpu;

    [FieldOffset(0x0010)] public IntPtr tensor_split;  //float* tensor_split;
    [FieldOffset(0x0018)] public llama_progress_callback progress_callback;
    [FieldOffset(0x0020)] public IntPtr progress_callback_user_data;

    [FieldOffset(0x0028)] public IntPtr kv_overrides;  //llama_model_kv_override* kv_overrides;
    [FieldOffset(0x0030)] public bool vocab_only;
    [FieldOffset(0x0031)] public bool use_mmap;
    [FieldOffset(0x0032)] public bool use_mlock;

    public static llama_model_params Default() =>
        new llama_model_params
        {
            n_gpu_layers = 0,
            main_gpu = 0,
            tensor_split = IntPtr.Zero,
            progress_callback = null,
            progress_callback_user_data = IntPtr.Zero,
            kv_overrides = IntPtr.Zero,
            vocab_only = false,
            use_mmap = true,
            use_mlock = false,
            split_mode = llama_split_mode.LLAMA_SPLIT_LAYER,
        };
}

public delegate bool llama_progress_callback(float progress);

