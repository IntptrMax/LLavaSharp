using System;
using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
public struct llama_token_data_array
{
    public IntPtr data;
    public ulong size;
    public bool sorted;
}