using System;
using clip_vision_model = System.IntPtr;
using ggml_allocr = System.IntPtr;
using ggml_backend_buffer_t = System.IntPtr;
using ggml_backend_t = System.IntPtr;
using ggml_context = System.IntPtr;
using gguf_context = System.IntPtr;
using int32_t = System.Int32;
using uint8_t = System.Byte;
public class clip_ctx
{
    public bool has_text_encoder = false;
    public bool has_vision_encoder = false;
    public bool has_llava_projector = false;

    public clip_vision_model vision_model = IntPtr.Zero;
    public projector_type proj_type = projector_type.PROJECTOR_TYPE_MLP;

    public float[] image_mean = new float[3];
    public float[] image_std = new float[3];
    public bool use_gelu = false;
    public int32_t ftype = 1;

    public gguf_context ctx_gguf;
    public ggml_context ctx_data;

    public uint8_t[] buf_compute_meta;

    // memory buffers to evaluate the model
    public ggml_backend_buffer_t params_buffer = IntPtr.Zero;
    public ggml_backend_buffer_t compute_buffer = IntPtr.Zero;
    public ggml_backend_t backend = IntPtr.Zero;
    public ggml_allocr compute_alloc = IntPtr.Zero;
}
