"""Replay of epoched trials as a timed window stream, plus posterior caching.

This is the last subpackage on the decoder side of the pipeline. Everything
downstream reads the `.npz` files written here and never touches EEG again.
"""
