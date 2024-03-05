public enum llama_split_mode
{
    LLAMA_SPLIT_NONE = 0, // single GPU
    LLAMA_SPLIT_LAYER = 1, // split layers and KV across GPUs
    LLAMA_SPLIT_ROW = 2, // split rows across GPUs
};