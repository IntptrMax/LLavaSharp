using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using llama_token = System.Int32;

namespace LLavaSharp
{
    public class Lib
    {
        public static llava_context llava_init(gpt_params @params)
        {
            Native.ggml_time_init();
            string clip_model_path = @params.mmproj;
            if (string.IsNullOrEmpty(@params.prompt))
            {
                @params.prompt = "describe the image in detail.";
            }

            IntPtr ctx_clip = Native.clip_model_load(clip_model_path);
            if (IntPtr.Zero == ctx_clip)
            {
                throw new Exception("error: unable to load clip_model\n");
            }
            Native.llama_backend_init(@params.numa);

            llama_model_params model_params = llama_model_params_from_gpt_params(@params);
            IntPtr model = Native.llama_load_model_from_file(@params.model, model_params);
            if (IntPtr.Zero == model)
            {
                throw new Exception("error: failed to load the llama_model\n");
            }
            llama_context_params ctx_params = llama_context_params_from_gpt_params(@params);
            ctx_params.n_ctx = @params.n_ctx < 4096 ? 4096 : @params.n_ctx; // we need a longer context size to process image embeddings, and if in llava-1.6 at least image embedding size (2880 tokens) + batch size (512). thanks to @zsogitbe
            IntPtr ctx_llama = Native.llama_new_context_with_model(model, ctx_params);

            if (IntPtr.Zero == ctx_llama)
            {
                throw new Exception("error: failed to create the llama_context\n");
            }
            llava_context ctx_llava = new llava_context();

            ctx_llava.ctx_llama = ctx_llama;
            ctx_llava.ctx_clip = ctx_clip;
            ctx_llava.model = model;
            return ctx_llava;
        }


        private static llama_model_params llama_model_params_from_gpt_params(gpt_params @params)
        {
            llama_model_params mparams = llama_model_params.Default();

            if (@params.n_gpu_layers != -1)
            {
                mparams.n_gpu_layers = @params.n_gpu_layers;
            }
            mparams.main_gpu = @params.main_gpu;
            mparams.split_mode = @params.split_mode;
            mparams.tensor_split = @params.tensor_split;
            mparams.use_mmap = @params.use_mmap;
            mparams.use_mlock = @params.use_mlock;
            if (null == @params.kv_overrides)
            {
                mparams.kv_overrides = new llama_model_kv_override[] { };
            }
            else
            {
                //GGML_ASSERT(@params.kv_overrides.back().key[0] == 0 && "KV overrides not terminated with empty key");
                mparams.kv_overrides = @params.kv_overrides;
            }

            return mparams;
        }

        private static llama_context_params llama_context_params_from_gpt_params(gpt_params @params)
        {
            llama_context_params cparams = new llama_context_params();

            cparams.n_ctx = @params.n_ctx;
            cparams.n_batch = @params.n_batch;
            cparams.n_threads = @params.n_threads;
            cparams.n_threads_batch = @params.n_threads_batch == -1 ? @params.n_threads : @params.n_threads_batch;
            cparams.mul_mat_q = @params.mul_mat_q;
            cparams.seed = @params.seed;
            cparams.logits_all = @params.logits_all;
            cparams.embedding = @params.embedding;
            cparams.rope_scaling_type = @params.rope_scaling_type;
            cparams.rope_freq_base = @params.rope_freq_base;
            cparams.rope_freq_scale = @params.rope_freq_scale;
            cparams.yarn_ext_factor = @params.yarn_ext_factor;
            cparams.yarn_attn_factor = @params.yarn_attn_factor;
            cparams.yarn_beta_fast = @params.yarn_beta_fast;
            cparams.yarn_beta_slow = @params.yarn_beta_slow;
            cparams.yarn_orig_ctx = @params.yarn_orig_ctx;
            cparams.offload_kqv = !@params.no_kv_offload;

            cparams.type_k = kv_cache_type_from_str(@params.cache_type_k);
            cparams.type_v = kv_cache_type_from_str(@params.cache_type_v);

            return cparams;
        }

        private static ggml_type kv_cache_type_from_str(string s)
        {
            if (s == "f32")
            {
                return ggml_type.GGML_TYPE_F32;
            }
            if (s == "f16")
            {
                return ggml_type.GGML_TYPE_F16;
            }
            if (s == "q8_0")
            {
                return ggml_type.GGML_TYPE_Q8_0;
            }
            if (s == "q4_0")
            {
                return ggml_type.GGML_TYPE_Q4_0;
            }
            if (s == "q4_1")
            {
                return ggml_type.GGML_TYPE_Q4_1;
            }
            if (s == "q5_0")
            {
                return ggml_type.GGML_TYPE_Q5_0;
            }
            if (s == "q5_1")
            {
                return ggml_type.GGML_TYPE_Q5_1;
            }

            throw new Exception("Invalid cache type: " + s);
        }

        public static llava_image_embed load_image(llava_context ctx_llava, gpt_params @params)
        {
            byte[] bytes = System.IO.File.ReadAllBytes(@params.image);
            return Native.llava_image_embed_make_with_bytes(ctx_llava.ctx_clip, @params.n_threads, bytes, bytes.Length);
            //IntPtr intptr_to_image_embed = Native.llava_image_embed_make_with_filename(ctx_llava.ctx_clip, @params.n_threads, @params.image);
            //llava_image_embed image_embed = (llava_image_embed)Marshal.PtrToStructure(intptr_to_image_embed, typeof(llava_image_embed));
            //return image_embed;
        }

        public static llava_image_embed load_image(llava_context ctx_llava, Bitmap bitmap, int n_threads)
        {
            using (MemoryStream stream = new MemoryStream())
            {
                bitmap.Save(stream, System.Drawing.Imaging.ImageFormat.Bmp);

                byte[] bytes = stream.ToArray();
                return Native.llava_image_embed_make_with_bytes(ctx_llava.ctx_clip, n_threads, bytes, bytes.Length);
            }
        }

        private static bool llama_should_add_bos_token(IntPtr model)
        {
            bool add_bos = Native.llama_add_bos_token(model);
            return add_bos ? add_bos : (Native.llama_vocab_type(model) == llama_vocab_type.LLAMA_VOCAB_TYPE_SPM);
        }

        private static bool eval_string(IntPtr ctx_llama, string str, int n_batch, ref int n_past, bool add_bos)
        {
            string str2 = str;
            IntPtr model = Native.llama_get_model(ctx_llama);
            int[] embd_inp = llama_tokenize(model, str2, add_bos, true);
            eval_tokens(ctx_llama, embd_inp, n_batch, ref n_past);
            return true;
        }

        private static int[] llama_tokenize(IntPtr model, string text, bool add_bos = false, bool special = true)
        {
            // upper limit for the number of tokens
            int n_tokens = text.Length + (add_bos ? 1 : 0);
            int[] result = new int[n_tokens];

            n_tokens = Native.llama_tokenize(model, text, text.Length, result, result.Length, add_bos, special);
            Array.Resize(ref result, Math.Abs(n_tokens));
            return result;
        }

        private static bool eval_tokens(IntPtr ctx_llama, int[] tokens, int n_batch, ref int n_past)
        {
            int N = tokens.Length;
            for (int i = 0; i < N; i += n_batch)
            {
                int n_eval = tokens.Length - i;
                if (n_eval > n_batch)
                {
                    n_eval = n_batch;
                }
                IntPtr intPtr = Marshal.UnsafeAddrOfPinnedArrayElement(tokens, i);
                //llama_batch batch = Native.llama_batch_get_one(intPtr, n_eval, ref n_past, 0);
                llama_batch batch = llama_batch_get_one(intPtr, n_eval, ref n_past, 0);

                int num = (int)Marshal.PtrToStructure(batch.token, typeof(int));

                int decodeResult = Native.llama_decode(ctx_llama, batch);
                if (0 != decodeResult)
                {
                    Console.WriteLine("failed to eval token\n");
                    return false;
                }
                n_past += n_eval;
            }
            return true;
        }

        private static bool llava_eval_image_embed(IntPtr ctx_llama, llava_image_embed image_embed, int n_batch, ref int n_past)
        {
            int n_embd = Native.llama_n_embd(Native.llama_get_model(ctx_llama));

            for (int i = 0; i < image_embed.n_image_pos; i += n_batch)
            {
                int n_eval = image_embed.n_image_pos - i;
                if (n_eval > n_batch)
                {
                    n_eval = n_batch;
                }
                llama_batch batch = new llama_batch
                {
                    n_tokens = n_eval,
                    token = IntPtr.Zero,
                    embd = image_embed.embed + i * n_embd,
                    pos = IntPtr.Zero,
                    n_seq_id = IntPtr.Zero,
                    seq_id = IntPtr.Zero,
                    logits = IntPtr.Zero,
                    all_pos_0 = n_past,
                    all_pos_1 = 1,
                    all_seq_id = 0

                };
                if (0 != Native.llama_decode(ctx_llama, batch))
                {
                    throw new Exception("failed to eval image embed\n");
                }
                n_past += n_eval;
            }
            return true;
        }

        private static llama_batch llama_batch_get_one(IntPtr tokens, int n_tokens, ref int pos_0, int seq_id)
        {
            return new llama_batch
            {
                n_tokens = n_tokens,
                token = tokens,
                embd = IntPtr.Zero,
                pos = IntPtr.Zero,
                n_seq_id = IntPtr.Zero,
                seq_id = IntPtr.Zero,
                logits = IntPtr.Zero,
                all_pos_0 = pos_0,
                all_pos_1 = 1,
                all_seq_id = seq_id
            };
        }

        public static string process_prompt(llava_context ctx_llava, llava_image_embed image_embed, gpt_params @params, string prompt, float temp = 0)
        {
            string QUESTION_ANSWERING_PROMPT = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, brief, and polite answers to the human's questions.\nUSER:";
            string ASSISTANT_PROMPT_SUFFIX = "\nASSISTANT:";

            int n_past = 0;
            int max_tgt_len = @params.n_predict < 0 ? 256 : @params.n_predict;
            //IntPtr model = Native.llama_get_model(ctx_llava.ctx_llama);
            bool add_bos = Lib.llama_should_add_bos_token(ctx_llava.model);


            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Restart();
            // llava chat format is "<system_prompt>\nUSER:<image_embeddings>\n<textual_prompt>\nASSISTANT:"
            Lib.eval_string(ctx_llava.ctx_llama, QUESTION_ANSWERING_PROMPT, @params.n_batch, ref n_past, add_bos);
            Console.WriteLine("Time to eval system prompt: " + stopwatch.ElapsedMilliseconds);

            stopwatch.Restart();
            //Lib.llava_eval_image_embed(ctx_llava.ctx_llama, image_embed, @params.n_batch, ref n_past);
            Native.llava_eval_image_embed(ctx_llava.ctx_llama, image_embed, @params.n_batch, ref n_past);
            Console.WriteLine("Time to eval image embed: " + stopwatch.ElapsedMilliseconds);

            stopwatch.Restart();
            Lib.eval_string(ctx_llava.ctx_llama, prompt + ASSISTANT_PROMPT_SUFFIX, @params.n_batch, ref n_past, false);
            Console.WriteLine("Time to eval textual prompt: " + stopwatch.ElapsedMilliseconds);


            llama_sampling_context ctx_sampling = llama_sampling_init(@params.sparams);

            StringBuilder stringBuilder = new StringBuilder();
            for (int i = 0; i < max_tgt_len; i++)
            {
                string tmp = sample(ctx_sampling, ctx_llava.ctx_llama, ref n_past, temp);
                if (tmp == "</s>") break;
                if (tmp.Contains("###")) break; // Yi-VL behavior
                if (tmp.Contains("<|im_end|>")) break; // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
                if (tmp.Contains("<|im_start|>")) break; // Yi-34B llava-1.6
                if (tmp.Contains("USER:")) break; // mistral llava-1.6
                stringBuilder.Append(tmp);
                //Console.Write(tmp);
            }
            //llama_sampling_free(ctx_sampling);
            //GCHandle handle = GCHandle.Alloc(ctx_sampling);
            //IntPtr ptr = GCHandle.ToIntPtr(handle);
            //Console.WriteLine();
            return stringBuilder.ToString();
        }

        private static llama_sampling_context llama_sampling_init(llama_sampling_params @params)
        {
            llama_sampling_context result = new llama_sampling_context();

            result.@params = @params;
            result.grammar = IntPtr.Zero;

            Array.Resize(ref result.prev, @params.n_prev);
            return result;
        }
        private static llama_token llama_sampling_sample(llama_sampling_context ctx_sampling, IntPtr ctx_main, IntPtr ctx_cfg, float temp = 0, int idx = 0)
        {
            // Call the implementation function with is_resampling set to false by default
            return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, false, temp);
        }

        private static string sample(llama_sampling_context ctx_sampling, IntPtr ctx_llama, ref int n_past, float temp = 0)
        {
            int id = llama_sampling_sample(ctx_sampling, ctx_llama, IntPtr.Zero, temp);
            llama_sampling_accept(ref ctx_sampling, id);
            string ret = string.Empty;
            if (id == Native.llama_token_eos(Native.llama_get_model(ctx_llama)))
            {
                ret = "</s>";
            }
            else
            {
                ret = llama_token_to_piece(ctx_llama, id);
            }
            eval_id(ctx_llama, id, ref n_past);
            return ret;
        }

        private static bool eval_id(IntPtr ctx_llama, int id, ref int n_past)
        {
            int[] tokens = new int[] { id };
            return Lib.eval_tokens(ctx_llama, tokens, 1, ref n_past);
        }

        private static int llama_sampling_sample_impl(llama_sampling_context ctx_sampling, IntPtr ctx_main, IntPtr ctx_cfg, int idx, bool is_resampling, float temp = 0)
        {  // Add a parameter to indicate if we are resampling
            llama_sampling_params @params = ctx_sampling.@params;

            int n_vocab = Native.llama_n_vocab(Native.llama_get_model(ctx_main));

            //float temp = @params.temp;
            temp = (temp == 0 ? @params.temp : temp);
            int penalty_last_n = @params.penalty_last_n < 0 ? @params.n_prev : @params.penalty_last_n;
            float penalty_repeat = @params.penalty_repeat;
            float penalty_freq = @params.penalty_freq;
            float penalty_present = @params.penalty_present;
            int mirostat = @params.mirostat;
            float mirostat_tau = @params.mirostat_tau;
            float mirostat_eta = @params.mirostat_eta;
            bool penalize_nl = @params.penalize_nl;

            llama_token[] prev = ctx_sampling.prev;
            llama_token_data[] cur = ctx_sampling.cur;

            int id = 0;

            // Get a pointer to the logits
            IntPtr logits = Native.llama_get_logits_ith(ctx_main, idx);
            int logitsLength = Native.llama_n_vocab(Native.llama_get_model(ctx_main));
            float[] original_logits = new float[logitsLength];

            // Declare original_logits at the beginning of the function scope

            if (!is_resampling)
            {
                int original_logits_length = Native.llama_n_vocab(Native.llama_get_model(ctx_main));
                original_logits = new float[original_logits_length];
                // Only make a copy of the original logits if we are not in the resampling phase, not sure if I actually have to do this.
                Marshal.Copy(logits, original_logits, 0, original_logits_length);
            }

            cur = new llama_token_data[n_vocab];

            for (int token_id = 0; token_id < n_vocab; token_id++)
            {
                cur[token_id] = new llama_token_data { id = token_id, logit = original_logits[token_id], p = 0.0f };
            }

            IntPtr cur_data = Marshal.UnsafeAddrOfPinnedArrayElement(cur, 0);
            llama_token_data_array cur_p = new llama_token_data_array { data = cur_data, size = (ulong)cur.Length, sorted = false };

            //// apply penalties
            //llama_token[] penalty_tokens = @params.use_penalty_prompt_tokens ? @params.penalty_prompt_tokens : prev;

            //IntPtr penalty_tokens_data = Marshal.UnsafeAddrOfPinnedArrayElement(penalty_tokens, 0);
            //int penalty_tokens_used_size = Math.Min(penalty_tokens.Length, penalty_last_n);
            //if (0 != penalty_tokens_used_size)
            //{
            //    //float nl_logit = original_logits[Native.llama_token_nl(Native.llama_get_model(ctx_main))];

            //    Native.llama_sample_repetition_penalties(ctx_main, ref cur_p,
            //         penalty_tokens_data + penalty_tokens.Length - penalty_tokens_used_size,
            //         penalty_tokens_used_size, penalty_repeat, penalty_freq, penalty_present);
            //}

            StringBuilder stringBuilder = new StringBuilder();
            for (int i = 0; i < 32000; i++)
            {
                llama_token_data dt = (llama_token_data)Marshal.PtrToStructure(Marshal.UnsafeAddrOfPinnedArrayElement(cur, i), typeof(llama_token_data));
                stringBuilder.AppendLine($"{dt.logit}");
            }
            File.WriteAllText("dt.txt", stringBuilder.ToString());

            int min_keep = Math.Max(1, @params.n_probs);

            sampler_queue(ctx_main, @params, ref cur_p, min_keep);


            id = Native.llama_sample_token(ctx_main, ref cur_p);

            return id;
        }

        private static void sampler_queue(IntPtr ctx_main, llama_sampling_params @params, ref llama_token_data_array cur_p, int min_keep)
        {
            int n_vocab = Native.llama_n_vocab(Native.llama_get_model(ctx_main));

            float temp = @params.temp;
            float dynatemp_range = @params.dynatemp_range;
            float dynatemp_exponent = @params.dynatemp_exponent;
            int top_k = @params.top_k <= 0 ? n_vocab : @params.top_k;
            float top_p = @params.top_p;
            float min_p = @params.min_p;
            float tfs_z = @params.tfs_z;
            float typical_p = @params.typical_p;
            string samplers_sequence = @params.samplers_sequence;

            foreach (char s in samplers_sequence)
            {
                switch (s)
                {
                    case 'k': Native.llama_sample_top_k(ctx_main, ref cur_p, top_k, min_keep); break;
                    case 'f': Native.llama_sample_tail_free(ctx_main, ref cur_p, tfs_z, min_keep); break;
                    case 'y': Native.llama_sample_typical(ctx_main, ref cur_p, typical_p, min_keep); break;
                    case 'p': Native.llama_sample_top_p(ctx_main, ref cur_p, top_p, min_keep); break;
                    case 'm': Native.llama_sample_min_p(ctx_main, ref cur_p, min_p, min_keep); break;
                    case 't':
                        if (dynatemp_range > 0)
                        {
                            float dynatemp_min = Math.Max(0.0f, temp - dynatemp_range);
                            float dynatemp_max = Math.Max(0.0f, temp + dynatemp_range);
                            Native.llama_sample_entropy(ctx_main, ref cur_p, dynatemp_min, dynatemp_max, dynatemp_exponent);
                        }
                        else
                        {
                            Native.llama_sample_temp(ctx_main, ref cur_p, temp);
                        }
                        break;
                    default: break;
                }
            }

        }

        private static void llama_sampling_accept(ref llama_sampling_context ctx_sampling, llama_token id)
        {
            List<llama_token> prev = ctx_sampling.prev.ToList();
            prev.RemoveAt(0);
            prev.Add(id);
            ctx_sampling.prev = prev.ToArray();
        }

        private static string llama_token_to_piece(IntPtr ctx, llama_token token)
        {
            char[] result = new char[8];
            int n_tokens = Native.llama_token_to_piece(Native.llama_get_model(ctx), token, result, result.Length);
            if (n_tokens < 0)
            {
                Array.Resize(ref result, -n_tokens);
                Native.llama_token_to_piece(Native.llama_get_model(ctx), token, result, result.Length);
            }
            else
            {
                Array.Resize(ref result, n_tokens);
            }
            //return new string(result);
            string str = Encoding.UTF8.GetString(Encoding.UTF8.GetBytes(new string(result, 0, Math.Abs(n_tokens))));

            return str;
        }

        public static void llama_print_timings(llava_context ctx)
        {
            Native.llama_print_timings(ctx);
        }

        public static void llava_image_embed_free(llava_image_embed embed)
        {
            Marshal.FreeHGlobal(embed.embed);
            embed = null;
        }

        public static void llava_free(llava_context ctx_llava)
        {
            if (IntPtr.Zero != ctx_llava.ctx_clip)
            {
                Native.clip_free(ctx_llava.ctx_clip);
                ctx_llava.ctx_clip = IntPtr.Zero;
            }

            Native.llama_free(ctx_llava.ctx_llama);
            Native.llama_free_model(ctx_llava.model);
            Native.llama_backend_free();
        }

        public static void llama_free_kv_cache(IntPtr llama_ctx)
        {
            Native.llama_kv_cache_clear(llama_ctx);
        }

        public static IntPtr LoadLibrary(string libraryPath)
        {
            return Native.LoadLibrary(libraryPath);
        }

        public static bool FreeLibrary(IntPtr libraryPtr)
        {
            return Native.FreeLibrary(libraryPtr);
        }
    }
}

