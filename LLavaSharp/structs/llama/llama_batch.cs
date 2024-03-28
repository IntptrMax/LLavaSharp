using System;
using System.Runtime.InteropServices;
using int32_t = System.Int32;
using llama_pos = System.Int32;
using llama_seq_id = System.Int32;

/// <summary>
/// Input data for llama_decode. 
/// A llama_batch object can contain input about one or many sequences. 
/// The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
/// </summary>
[StructLayout(LayoutKind.Explicit, Pack = 8)]
struct llama_batch
{

    [FieldOffset(0x0000)] public int32_t n_tokens;

    /// <summary>
    /// the token ids of the input (used when embd is NULL)
    /// </summary>
    [FieldOffset(0x0008)] public IntPtr token;

    /// <summary>
    /// token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    /// </summary>
    [FieldOffset(0x0010)] public IntPtr embd;

    /// <summary>
    /// the positions of the respective token in the sequence
    /// </summary>
    [FieldOffset(0x0018)] public IntPtr pos;

    /// <summary>
    /// the sequence to which the respective token belongs
    /// </summary>
    [FieldOffset(0x0020)] public IntPtr n_seq_id;
    [FieldOffset(0x028)] public IntPtr seq_id;

    /// <summary>
    /// if zero, the logits (and/or the embeddings) for the respective token will not be output
    /// </summary>
    [FieldOffset(0x0030)] public IntPtr logits;

    // NOTE: helpers for smooth API transition - can be deprecated in the future
    //       for future-proof code, use the above fields instead and ignore everything below
    //
    // pos[i] = all_pos_0 + i*all_pos_1
    //

    /// <summary>
    /// used if pos == NULL
    /// </summary>
    [FieldOffset(0x0038)] public llama_pos all_pos_0;

    /// <summary>
    /// used if pos == NULL
    /// </summary>
    [FieldOffset(0x0038 + 4)] public llama_pos all_pos_1;

    /// <summary>
    /// used if seq_id == NULL
    /// </summary>
    [FieldOffset(0x0040)] public llama_seq_id all_seq_id;
}

