using System;
using System.Collections.Generic;
using int32_t = System.Int32;
using int8_t = System.SByte;
using size_t = System.UInt64;

public class gpt_params
{
    const int LLAMA_ROPE_SCALING_UNSPECIFIED = -1;
    const int LLAMA_MAX_DEVICES = 1;

    public int32_t seed;    // RNG seed

    public int32_t n_threads = get_num_physical_cores();
    public int32_t n_threads_batch = -1;    // number of threads to use for batch processing (-1 = use n_threads)
    public int32_t n_predict = -1;    // new tokens to predict
    public int32_t n_ctx = 512;   // context size
    public int32_t n_batch = 512;   // batch size for prompt processing (must be >=32 to use BLAS)
    public int32_t n_keep = 0;     // number of tokens to keep from initial prompt
    public int32_t n_draft = 8;     // number of tokens to draft during speculative decoding
    public int32_t n_chunks = -1;    // max number of chunks to process (-1 = unlimited)
    public int32_t n_parallel = 1;     // number of parallel sequences to decode
    public int32_t n_sequences = 1;     // number of sequences to decode
    public float p_accept = 0.5f;  // speculative decoding accept probability
    public float p_split = 0.1f;  // speculative decoding split probability
    public int32_t n_gpu_layers = -1;    // number of layers to store in VRAM (-1 - use default)
    public int32_t n_gpu_layers_draft = -1;    // number of layers to store in VRAM for the draft model (-1 - use default)
    public llama_split_mode split_mode = llama_split_mode.LLAMA_SPLIT_LAYER; // how to split the model across GPUs
    public int32_t main_gpu = 0;     // the GPU that is used for scratch and small tensors
    public float[] tensor_split = new float[LLAMA_MAX_DEVICES] { 0 };   // how split tensors should be distributed across GPUs
    public int32_t n_beams = 0;     // if non-zero then use beam search of given width.
    public int32_t grp_attn_n = 1;     // group-attention factor
    public int32_t grp_attn_w = 512;   // group-attention width
    public float rope_freq_base = 0.0f;  // RoPE base frequency
    public float rope_freq_scale = 0.0f;  // RoPE frequency scaling factor
    public float yarn_ext_factor = -1.0f; // YaRN extrapolation mix factor
    public float yarn_attn_factor = 1.0f;  // YaRN magnitude scaling factor
    public float yarn_beta_fast = 32.0f; // YaRN low correction dim
    public float yarn_beta_slow = 1.0f;  // YaRN high correction dim
    public int32_t yarn_orig_ctx = 0;     // YaRN original context length
    public int8_t rope_scaling_type = LLAMA_ROPE_SCALING_UNSPECIFIED; // TODO: better to be int32_t for alignment
                                                                      //       pinging @cebtenzzre

    // // sampling parameters
    public llama_sampling_params sparams = new llama_sampling_params();

    public string model = "models/7B/ggml-model-f16.gguf"; // model path
    public string model_draft = "";                              // draft model for speculative decoding
    public string model_alias = "unknown"; // model alias
    public string prompt = "";
    public string prompt_file = "";  // store the external prompt file name
    public string path_prompt_cache = "";  // path to file for saving/loading prompt eval state
    public string input_prefix = "";  // string to prefix user inputs with
    public string input_suffix = "";  // string to suffix user inputs with
    public string[] antiprompt = new string[] { }; // string upon seeing which more user input is prompted
    public string logdir = "";  // directory in which to save YAML log files

    public llama_model_kv_override[] kv_overrides;

    // TODO: avoid tuple, use struct
    public List<Tuple<string, float>> lora_adapter = new List<Tuple<string, float>>(); // lora adapter path with user defined scale
    public string lora_base = "";                              // base model path for the lora adapter

    public int ppl_stride = 0;     // stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
    public int ppl_output_type = 0;     // = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line
                                        //                                       (which is more convenient to use for plotting)
                                        //
    public bool hellaswag = false; // compute HellaSwag score over random tasks from datafile supplied in prompt
    public size_t hellaswag_tasks = 400;   // number of tasks to use when computing the HellaSwag score

    public bool mul_mat_q = true;  // if true, use mul_mat_q kernels instead of cuBLAS
    public bool random_prompt = false; // do not randomize prompt if none provided
    public bool use_color = false; // use color to distinguish generations and inputs
    public bool interactive = false; // interactive mode
    public bool chatml = false; // chatml mode (used for models trained on chatml syntax)
    public bool prompt_cache_all = false; // save user input and generations to prompt cache
    public bool prompt_cache_ro = false; // open the prompt cache read-only and do not update it

    public bool embedding = false; // get only sentence embedding
    public bool escape = false; // escape "\n", "\r", "\t", "\'", "\"", and "\\"
    public bool interactive_first = false; // wait for user input immediately
    public bool multiline_input = false; // reverse the usage of `\`
    public bool simple_io = false; // improves compatibility with subprocesses and limited consoles
    public bool cont_batching = false; // insert new sequences for decoding on-the-fly

    public bool input_prefix_bos = false; // prefix BOS to user inputs, preceding input_prefix
    public bool ignore_eos = false; // ignore generated EOS tokens
    public bool instruct = false; // instruction mode (used for Alpaca models)
    public bool logits_all = false; // return logits for all tokens in the batch
    public bool use_mmap = true;  // use mmap for faster loads
    public bool use_mlock = false; // use mlock to keep model in memory
    public bool numa = false; // attempt optimizations that help on some NUMA systems
    public bool verbose_prompt = false; // print prompt tokens before generation
    public bool infill = false; // use infill mode
    public bool dump_kv_cache = false; // dump the KV cache contents for debugging purposes
    public bool no_kv_offload = false; // disable KV offloading

    public string cache_type_k = "f16"; // KV cache data type for the K
    public string cache_type_v = "f16"; // KV cache data type for the V

    // multimodal models (see examples/llava)
    public string mmproj = ""; // path to multimodal projector
    public string image = ""; // path to an image file


    private static int get_num_physical_cores()
    {
        int n_cores = Environment.ProcessorCount;
        return n_cores > 0 ? (n_cores <= 4 ? n_cores : n_cores / 2) : 4;
    }

};



