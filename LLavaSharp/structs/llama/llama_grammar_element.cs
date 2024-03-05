using uint32_t = System.UInt32;
public struct llama_grammar_element
{
    public llama_gretype type;
    public uint32_t value; // Unicode code point or rule ID
}

public enum llama_gretype
{
    // end of rule definition
    LLAMA_GRETYPE_END = 0,

    // start of alternate definition for rule
    LLAMA_GRETYPE_ALT = 1,

    // non-terminal element: reference to rule
    LLAMA_GRETYPE_RULE_REF = 2,

    // terminal element: character (code point)
    LLAMA_GRETYPE_CHAR = 3,

    // inverse char(s) ([^a], [^a-b] [^abc])
    LLAMA_GRETYPE_CHAR_NOT = 4,

    // modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to
    // be an inclusive range ([a-z])
    LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,

    // modifies a preceding LLAMA_GRETYPE_CHAR or
    // LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
    LLAMA_GRETYPE_CHAR_ALT = 6,
};

