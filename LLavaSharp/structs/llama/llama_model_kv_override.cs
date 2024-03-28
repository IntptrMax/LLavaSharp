using System.Runtime.InteropServices;
using int64_t = System.Int64;

[StructLayout(LayoutKind.Explicit, Pack = 8)]
public class llama_model_kv_override
{
    [FieldOffset(0x0000)] byte[] key = new byte[128];
    [FieldOffset(0x0080)] llama_model_kv_override_type tag;
    [FieldOffset(0x0088)] public union union;
}

[StructLayout(LayoutKind.Explicit, Size = 8)]
public struct union
{
    [FieldOffset(0)] public int64_t int_value;
    [FieldOffset(0)] public double float_value;
    [FieldOffset(0)] public bool bool_value;
};

