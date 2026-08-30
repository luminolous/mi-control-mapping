"""Control mappings: posterior plus robot state to a velocity command.

Must not import `micm.decoders` or `micm.data`. Mappings read cached posterior
files, which is what makes iterating on a mapping cheap.
"""
