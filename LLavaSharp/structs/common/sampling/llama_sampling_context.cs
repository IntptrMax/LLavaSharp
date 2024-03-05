using llama_token = System.Int32;
using llama_grammar = System.IntPtr;
using System.Runtime.InteropServices;
using System;

[StructLayout(LayoutKind.Sequential)]
public class llama_sampling_context
{
    // parameters that will be used for sampling
    public llama_sampling_params @params;

    // mirostat sampler state
    public float mirostat_mu = 0.0f;

    public llama_grammar grammar;

    // internal
    public IntPtr parsed_grammar;

    // TODO: replace with ring-buffer
    public llama_token[] prev;
    public llama_token_data[] cur;
}