using LLavaSharp;
using System;
using System.Drawing;

namespace LLavaSharpTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            IntPtr llamaDllPtr = Lib.LoadLibrary(@".\dll\cuda12\llama.dll");
            IntPtr llavaSharedDllPtr = Lib.LoadLibrary(@".\dll\cuda12\llava_shared.dll");

            gpt_params @params = new gpt_params();

            @params.image = @".\Image\1.jpg";
            @params.model = @".\models\model.gguf";
            @params.mmproj = @".\models\mmproj.gguf";

            @params.prompt = "describe the image.";
            @params.n_gpu_layers = 32;
            @params.n_gpu_layers_draft = 32;

            llava_context ctx_llava = Lib.llava_init(@params);

            llava_image_embed image_embed = Lib.load_image(ctx_llava, @params);
            string resuntl = Lib.process_prompt(ctx_llava, image_embed, @params, @params.prompt);
            Console.WriteLine(resuntl);
            Console.WriteLine();
            Lib.llava_image_embed_free(image_embed);
            Lib.llama_free_kv_cache(ctx_llava.ctx_llama); //need to free the context cache if want to prompt more images. thanks to @Rinne 
            //Lib.llama_print_timings(ctx_llava);
            GC.Collect();

            Lib.llava_free(ctx_llava);
            Console.WriteLine();
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();


            //HelperTest();
            //Console.WriteLine();
            //Console.WriteLine("Press any key to exit...");
            //Console.ReadKey();
        }

        private static void HelperTest()
        {
            IntPtr llamaDllPtr = Lib.LoadLibrary(@".\dll\cuda12\llama.dll");
            IntPtr llavaSharedDllPtr = Lib.LoadLibrary(@".\dll\cuda12\llava_shared.dll");

            using (Helper helper = new Helper(@".\models\7B\model.gguf", @".\models\7B\mmproj.gguf"))
            {
                string[] files = System.IO.Directory.GetFiles(@".\Image");
                foreach (var file in files)
                {
                    using (Bitmap bitmap = new Bitmap(file))
                    {
                        string result = helper.ProcessImage(bitmap, "Describe the image in simple words.");
                        Console.ForegroundColor = ConsoleColor.DarkGreen;
                        Console.WriteLine(result);
                        Console.ResetColor();
                    }
                }
            }
        }
    }
}
