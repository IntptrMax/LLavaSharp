using System;
using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
public class llava_image_embed
{
    public IntPtr embed;
    public int n_image_pos;
};

