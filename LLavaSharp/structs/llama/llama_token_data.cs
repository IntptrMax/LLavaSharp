using System.Runtime.InteropServices;
using llama_token = System.Int32;

[StructLayout(LayoutKind.Sequential)]
public struct llama_token_data
{
    public llama_token id; // token id
    public float logit;    // log-odds of the token
    public float p;        // probability of the token
}

