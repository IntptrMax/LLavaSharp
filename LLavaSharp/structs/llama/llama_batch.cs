using System;
using System.Runtime.InteropServices;
using int32_t = System.Int32;
using llama_pos = System.Int32;
using llama_seq_id = System.Int32;

[StructLayout(LayoutKind.Sequential)]
struct llama_batch
{

    public int32_t n_tokens;

    public IntPtr token;
    public IntPtr embd;
    public IntPtr pos;
    public IntPtr n_seq_id;
    public IntPtr seq_id;
    public IntPtr logits;

    // NOTE: helpers for smooth API transition - can be deprecated in the future
    //       for future-proof code, use the above fields instead and ignore everything below
    //
    // pos[i] = all_pos_0 + i*all_pos_1
    //
    public llama_pos all_pos_0;  // used if pos == NULL
    public llama_pos all_pos_1;  // used if pos == NULL
    public llama_seq_id all_seq_id; // used if seq_id == NULL
}

