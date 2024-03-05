using System;
using System.Drawing;

namespace LLavaSharp
{
    public class Helper : IDisposable
    {
        private llava_context ctx_llava;
        gpt_params @params = new gpt_params();

        public Helper(string model_path, string mmproj_path, int ngl = 32)
        {
            @params.model = model_path;
            @params.mmproj = mmproj_path;
            @params.n_gpu_layers = ngl;
            @params.n_gpu_layers_draft = ngl;
            ctx_llava = Lib.llava_init(@params);
        }

        public string ProcessImage(Bitmap bitmap, string prompt, float temp = 0)
        {
            llava_image_embed image_embed = Lib.load_image(ctx_llava, bitmap, @params.n_threads);
            string result = Lib.process_prompt(ctx_llava, image_embed, @params, prompt, temp);
            Lib.llava_image_embed_free(image_embed);
            Lib.llama_free_kv_cache(ctx_llava.ctx_llama);
            GC.Collect();
            return result;
        }

        public void Dispose()
        {
            Lib.llava_free(ctx_llava);
        }

    }
}
