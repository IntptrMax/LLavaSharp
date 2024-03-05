using System.Collections.Generic;
using uint32_t = System.UInt32;

public struct parse_state
{
    public Dictionary<string, uint32_t> symbol_ids;
    public List<List<llama_grammar_element>> rules;

    public llama_grammar_element[] c_rules;
}

