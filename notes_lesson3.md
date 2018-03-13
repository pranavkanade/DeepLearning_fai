# Fast AI Lesson 3

pd.Dataframe look into this so that : 37:50


* Softmax 

As the last layer is fully connected layer we don’t use rely their.
-> hence we could have different output which could also contain -ve not.
-> So to use softmax we need all the activations in +ve format hence we do exp(x) on them.
-> Once we, calculate it we can pass them to soft max which spits out the probabilities for each of the classes

-> for multi labelled classification we use sigmoid as the activation function for last layer