"""Type definitions."""

from typing import Any, Mapping, Optional, Sequence, TypeVar, Union

"""
A tensor in any of the backends.

We do not define it explicitly to not require all the backends to be installed
and imported. The explicit definition would be:
```
Union[
  numpy.ndarray,
  tensorflow.Tensor,
  tensorflow.RaggedTensor,
  tensorflow.SparseTensor,
  tensorflow.IndexedSlices,
  jax.Array,
  jax.experimental.sparse.JAXSparse,
  torch.Tensor,
  keras.KerasTensor,
]
```
"""
Tensor = Any

TensorShape = Sequence[Optional[int]]

T = TypeVar("T")
Nested = Union[
    T,
    Sequence[Union[T, "Nested[T]"]],
    Mapping[str, Union[T, "Nested[T]"]],
]
