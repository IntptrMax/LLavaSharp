using System.Runtime.InteropServices;
using int64_t = System.Int64;


public class llama_model_kv_override
{
    byte[] key = new byte[128];
    llama_model_kv_override_type tag;
    union union;
}

[StructLayout(LayoutKind.Sequential)]
struct union
{
    int64_t int_value;
    double float_value;
    bool bool_value;
};

