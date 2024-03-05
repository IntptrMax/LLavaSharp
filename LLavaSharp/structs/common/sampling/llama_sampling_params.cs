using System;
using System.Runtime.InteropServices;
using int32_t = System.Int32;
using llama_token = System.Int32;

//[StructLayout(LayoutKind.Sequential)]
//public struct llama_sampling_params
//{
//    public int32_t n_prev;       // number of previous tokens to remember
//    public int32_t n_probs;        // if greater than 0, output the probabilities of top n_probs tokens.
//    public int32_t top_k;       // <= 0 to use vocab size
//    public float top_p;    // 1.0 = disabled
//    public float min_p;    // 0.0 = disabled
//    public float tfs_z;    // 1.0 = disabled
//    public float typical_p;    // 1.0 = disabled
//    public float temp;    // <= 0.0 to sample greedily, 0.0 to not output probabilities
//    public float dynatemp_range;  //0.0 = disabled
//    public float dynatemp_exponent;  // controls how entropy maps to temperature in dynamic temperature sampler
//    public int32_t penalty_last_n;       // last n tokens to penalize (0 = disable penalty, -1 = context size)
//    public float penalty_repeat;    // 1.0 = disabled
//    public float penalty_freq;    // 0.0 = disabled
//    public float penalty_present;    // 0.0 = disabled
//    public int32_t mirostat;        // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
//    public float mirostat_tau;    // target entropy
//    public float mirostat_eta;    // learning rate
//    public bool penalize_nl;     // consider newlines as a repeatable token

//    public string samplers_sequence; // top_k, tail_free, typical_p, top_p, min_p, temp

//    public string grammar;  // optional BNF-like grammar to constrain sampling

//    // Classifier-Free Guidance
//    // https://arxiv.org/abs/2306.17806

//    public string cfg_negative_prompt; // string to help guidance
//    public float cfg_scale; // how strong is guidance

//    //std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens
//    public IntPtr logit_bias;
//    public IntPtr penalty_prompt_tokens;
//    public bool use_penalty_prompt_tokens;

//    public static llama_sampling_params Default => new llama_sampling_params
//    {
//        n_prev = 64,
//        n_probs = 0,
//        top_k = 40,
//        top_p = 0.95f,
//        min_p = 0.05f,
//        tfs_z = 1.00f,
//        typical_p = 1.00f,
//        temp = 0.80f,
//        dynatemp_range = 0.0f,
//        dynatemp_exponent = 1.0f,
//        penalty_last_n = 64,
//        penalty_repeat = 1.10f,
//        penalty_freq = 0.00f,
//        penalty_present = 0.00f,
//        mirostat = 0,
//        mirostat_tau = 5.00f,
//        mirostat_eta = 0.10f,
//        penalize_nl = true,
//        samplers_sequence = "kfypmt",
//        cfg_negative_prompt = string.Empty,
//        grammar = string.Empty,

//        cfg_scale = 1.0f, 
//        use_penalty_prompt_tokens = false,
//    };
//}
//public struct logit_bias_struct
//{
//    public llama_token token;
//    public float bias;
//}

[StructLayout(LayoutKind.Sequential)]
public class llama_sampling_params
{
    public int32_t n_prev = 64;       // number of previous tokens to remember
    public int32_t n_probs = 0;        // if greater than 0, output the probabilities of top n_probs tokens.
    public int32_t top_k = 40;       // <= 0 to use vocab size
    public float top_p = 0.95f;    // 1.0 = disabled
    public float min_p = 0.05f;    // 0.0 = disabled
    public float tfs_z = 1.00f;    // 1.0 = disabled
    public float typical_p = 1.00f;    // 1.0 = disabled
    public float temp = 0.70f;    // <= 0.0 to sample greedily, 0.0 to not output probabilities
    public float dynatemp_range = 0.0f;  //0.0 = disabled
    public float dynatemp_exponent = 1.0f;  // controls how entropy maps to temperature in dynamic temperature sampler
    public int32_t penalty_last_n = 64;       // last n tokens to penalize (0 = disable penalty, -1 = context size)
    public float penalty_repeat = 1.10f;    // 1.0 = disabled
    public float penalty_freq = 0.00f;    // 0.0 = disabled
    public float penalty_present = 0.00f;    // 0.0 = disabled
    public int32_t mirostat = 0;        // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    public float mirostat_tau = 5.00f;    // target entropy
    public float mirostat_eta = 0.10f;    // learning rate
    public bool penalize_nl = true;     // consider newlines as a repeatable token

    public string samplers_sequence = "kfypmt"; // top_k, tail_free, typical_p, top_p, min_p, temp

    public string grammar = string.Empty;  // optional BNF-like grammar to constrain sampling

    // Classifier-Free Guidance
    // https://arxiv.org/abs/2306.17806

    public string cfg_negative_prompt = string.Empty; // string to help guidance
    public float cfg_scale = 1.0f; // how strong is guidance

    //std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens
    public IntPtr logit_bias;
    public llama_token[] penalty_prompt_tokens;
    public bool use_penalty_prompt_tokens = false;
}
public struct logit_bias_struct
{
    public llama_token token;
    public float bias;
}



