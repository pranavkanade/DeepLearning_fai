# Fast AI Lesson 1

GPU renting ->
Crestel
Paperspace

Curl http://files.fast.ai/setup/paperspace | bash

Look for David Perkins

Universal approximation theorem => this kind of function can solve any given problems to arbitrarily close accuracy as long as you add enough parameters

Do this with gradient descent.

*	in case of neural nets there is always only one minima. There are no local minimas around. And we need to find the minima in shortest amount of time possible.

Deep learning simply means neural networks with multiple hidden layers.


Examples of deep learning => 
Google inbox auto reply 

Skype auto translator to Chinese or Spanish 

Look at paper => Semantic style transfer and turning two bit doodles into fine artworks.

*	Kernels

Linear models are followed by nonlineararity which intern can be used to build arbitrarily complex shapes 
Most of the times we use relu or sigmoid(not really but used to) 
We stay to adjust the promoters to fit the resultant shape to fit the solution.
And we need some kind of mechanism to do this => stochastic gradient descent

Explain how gradient descent works, actually dy/dx.

* You should avoid having too large of the step size as it actually will(may) lead to divergence rather than convergence. Which is bad. This is affected by the learning rate.

Formula => x(n+1) = x(n) + (dy/dx) * (L)			; (L) -> learning rate


What happens when you try to combine :
	
*	convolution
*	nonlinearity
*	gradient descent 

Paper -> Visualizing and understanding convolutional networks

Ultimately for the success it is important to set the learning rate to a reasonable value which will result in good accuracy without consuming infinite amount of time.

There is a paper on how to choose the learning rates and no one knows about it : 
Paper -> Cyclical learning rates for training neural networks.
This is implemented in fast ai library.  => learn.lr_find()

The approach is :
*	let set the learning rate to very small val.
*	after each step increase it 
*	what this will do is increase the loss function and then suddenly dropps
* The idea is to find the point where we saw the best impovement.
* We are looking for the point where this graph is dropping quickly. 
* We are not looking at the point where it was minimum coz it may be the point after which it suddenly started doing worst.

* The final thing is to decide the epoch. 
* Epoch :  going to our entire dataset ( going through all the mini batches ) 
* 

