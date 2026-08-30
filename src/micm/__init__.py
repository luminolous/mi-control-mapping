"""micm: control mapping versus decoder quality in motor-imagery BCI.

The package is split into two halves that communicate only through cached
posterior files on disk:

    data -> decoders -> replay -> [ .npz posterior files ]
                                          |
                                          v
                              mapping -> env -> eval -> viz

`micm.mapping` and `micm.env` must never import `micm.decoders` or
`micm.data`. That boundary is what keeps mapping iteration cheap: designing a
new mapping never re-runs a decoder. It is enforced by tests/test_architecture.py.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
