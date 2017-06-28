CMS\_Deep\_Learning\.layers
===================================


lorentz layer
-------------------------------------------

The Lorentz layer applies the relativistic Lorentz transformation to inputs of shape `(batch_size, N, 4)`, where each of N 4-vectors have a set of 3 weights corresponding to a boost relative to the lab frame. A boost matrix is found for each N sets of boost coordinates, and is applied to each of N 4-vectors independently. In short training resolves the coordinates of each boost, and each forward pass of the network applies the boost.

.. code-block:: python

    slicedA = Slice("[:,0:4]")(input)
    slicedB = Slice("[:,4:9]")(input)
    lorentzApplied = Lorentz(sphereCoords=False, vec_start=0)(slicedA)
	
The argument `sphereCoords` if set to True will find the weights in (theta, phi, mag) coordinates instead of Cartesian coordinates. It is up to the user to determine which coordinate system trains more effectively, but the Cartesian default has given better results in preliminary tests.

The argument `vec_start` determines where in the final dimension of the input to start reading each 4-vector. By default it reads from 0 (the 1st position) to 3 (the 4th positon). This is useful if one wants to forgo using slice layers and the 4-vector in the data is not held in the first 4 spots.

.. automodule:: CMS_Deep_Learning.layers.lorentz
    :members:
    :undoc-members:
    :show-inheritance:

slice layer
-----------------------------------------

The slice layer simply slices multidimentional data. For example to take an input of shape (batch_size, 100 , 9), and output two tensors with shape (batch_size, 100 , 4) and (batch_size, 100 , 5) a Slice layer can be used like so:

.. code-block:: python

	#input.shape is (batch_size, 100 , 9)
	slicedA = Slice("[:,0:4]")(input)
	slicedB = Slice("[:,4:9]")(input)

Taking note that the the user is protected from slicing on the dimension that hold the batch_size. So:

.. code-block:: python

	Slice("[:,0:10:2]")

is really applying `[:,:,0:10:2]` to the input tensor.

Each slice takes the form of `start:end:stepsize`, where `start:end` implies `start:end:1` and omitting start or end implies that the slice has a lower bound of 0 or the end of the axis respectively.

.. automodule:: CMS_Deep_Learning.layers.slice
    :members:
    :undoc-members:
    :show-inheritance:

