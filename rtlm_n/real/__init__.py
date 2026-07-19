"""Real-world RTLM-N: torch backbones, real text data, SGD baselines,
measured energy.

This subpackage requires torch (`pip install rtlm-n[real]`). The core
`rtlm_n` package stays NumPy-only; everything here plugs into the same
state algebra, node, merge-engine, and factory code paths - the point is
that moving from a toy backbone to a real one swaps the representation
extractor, not the intelligence-state machinery.
"""
