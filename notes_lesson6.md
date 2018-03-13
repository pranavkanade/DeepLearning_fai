* In learn object we can grab the pytorch model
`learn.model`
* These models are property - *basically a function but do not have to call it with the paranthesis*
* If we print down the models it'll show us which layers we have created.
* Models in pytorch need Variables and not tensors as they need to keep track of derivatives.
* Turning a tensor into variable
`V(tensor_var)`
* In fastai we have `to_np` this function returns a numpy array regardless of what you pass it. (tensor/variable)
* To bring every thing (model/ data) which has been processed by GPU on CPU just need to call `cpu()` on every thing.
__e.g.__ `learn.model.cpu()` or `V(topMovieIdx).cpu()`
