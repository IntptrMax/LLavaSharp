using System;
using System.Runtime.InteropServices;
using llama_pos = System.Int32;
using llama_seq = System.Int32;
using llama_token = System.Int32;
using size_t = System.UInt64;

namespace LLavaSharp
{
    internal class Native
    {
        const string llava_shared_dll = @"llava_shared";
        const string llama_dll = @"llama";
        const string kernel32_dll = "kernel32.dll";

        [DllImport(kernel32_dll, CharSet = CharSet.Auto, SetLastError = true)]
        public static extern IntPtr LoadLibrary(string libname);

        [DllImport(kernel32_dll, CharSet = CharSet.Auto, SetLastError = true)]
        public static extern bool FreeLibrary(IntPtr hModule);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public extern static void ggml_time_init();

        [DllImport(llava_shared_dll, CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr clip_model_load(string clip_model_path, int verbosity = 1);

        [DllImport(llava_shared_dll, CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr llava_image_embed_make_with_filename(IntPtr clip_ctx, int n_threads, string image);

        [DllImport(llava_shared_dll, CallingConvention = CallingConvention.Cdecl)]
        public extern static llava_image_embed llava_image_embed_make_with_bytes(IntPtr ctx_clip, int n_threads, byte[] bytes, int image_bytes_length);

        [DllImport(llava_shared_dll, CallingConvention = CallingConvention.Cdecl)]
        public extern static bool llava_eval_image_embed(IntPtr ctx_llama, llava_image_embed image_embed, int n_batch, ref int n_past);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public extern static void llama_backend_init(bool numa);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr llama_load_model_from_file(string model_path, llama_model_params lLamaModel);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr llama_new_context_with_model(IntPtr model, llama_context_params llama_Context_Params);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public extern static bool llama_add_bos_token(IntPtr model);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public extern static llama_vocab_type llama_vocab_type(IntPtr model);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public extern static int llama_tokenize(IntPtr model, string text, int text_len, [Out] int[] tokens, int n_max_tokens, bool add_bos, bool special);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public extern static llama_batch llama_batch_get_one(IntPtr tokens, int n_tokens, llama_pos pos_0, llama_seq seq_id);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public extern static int llama_decode(IntPtr ctx, llama_batch batch);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr llama_grammar_init(llama_grammar_element[] rules, size_t n_rules, size_t start_rule_index);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public extern static int llama_token_eos(IntPtr model);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_n_vocab(IntPtr model);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_get_logits_ith(IntPtr ctx, int i);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_token llama_token_nl(IntPtr model);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_repetition_penalties(IntPtr ctx, ref llama_token_data_array candidates, IntPtr last_tokens, int penalty_last_n, float penalty_repeat, float penalty_freq, float penalty_present);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_top_k(IntPtr ctx, ref llama_token_data_array candidates, int k, int min_keep);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_tail_free(IntPtr ctx, ref llama_token_data_array candidates, float z, int min_keep);
        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_typical(IntPtr ctx, ref llama_token_data_array candidates, float p, int min_keep);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_top_p(IntPtr ctx, ref llama_token_data_array candidates, float p, int min_keep);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_min_p(IntPtr ctx, ref llama_token_data_array candidates, float p, int min_keep);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_entropy(IntPtr ctx, ref llama_token_data_array candidates_p, float min_temp, float max_temp, float exponent_val);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_temp(IntPtr ctx, ref llama_token_data_array candidates_p, float temp);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_token_to_piece(IntPtr model, llama_token token, [Out] char[] buffer, int length);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_token llama_sample_token(IntPtr ctx, ref llama_token_data_array candidates);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_n_embd(IntPtr model);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_print_timings(llava_context ctx);

        [DllImport(llava_shared_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void clip_free(IntPtr ctx);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_free(IntPtr ctx);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_free_model(IntPtr model);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_backend_free();

        [DllImport(llava_shared_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llava_image_embed_free(IntPtr embed);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_kv_cache_clear(IntPtr llama_ctx);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_set_n_threads(IntPtr llama_context, uint n_threads, uint n_threads_batch);

        [DllImport(llama_dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_numa_init(ggml_numa_strategy numa);

    }
}
