* Metrix vectorization.
* Shallow learning.
* Matrix decomposition. => solved the movie rating prediction with xl. 
* Using a assumed vectors called as => latent factors
* Colaborative filtering using Probablistic matrics vectorization.
* #How do we find the dimension of the embedding vectorys ??#
* #SOL :# _we have no idea_ 
* weight decaying -> l2 regularization.


## This movie and this user -> 
### IF
1. Here are the same kind of movies this person likeed 
2. Here is the group of people who like this movie

### Conclusion 
1. which movies are similar in terms of similar ppl liked it 
2. which grp of ppl are similar to this person based on the people who liked the same kind of movie

#### Whenever this kind of underlying structure is needed colaborative filtering is the answer (or rather useful).

* dimentionality of the embedding metrix could be found by trying couple of examples.
* As there is no certain way to find it.
* n_factor in the lesson5 notebook is a diention of the embeddings.



# please look into regression vs classification.


* the dot product in fast ai

```python
a = T([[1, 2], [3, 4]])
b = T([[2, 2], [10, 10]])
# above are the 2D tensors from the pytorch

a*b         # this is the element wise multiplication of the above tensors
#   2   4
#   30  40

# the dot product
(a*b).sum(1)
#   6
#   70
```

* pytorch model -> we can through these entities inside a neural network. created on top of pytorch.

### Initializing the embeddings 
* use `kaiming he` algorithm



## BIAS

* If the movie is generally popular there is a high chance that all the ratings it gets will be avgly higher.
* Similarly some user might give higher ratings in general,
* This shows a pattern that there is a bias involved. which does not depend on any characteristic of movie or the person. but is attacted to a person or a movie. which is fized.


* please look in to #broadcasting#.
* please look at finite differencing checker.


#### MOMENTUM :
* when net hits towards the direction where it is making progress and somehow it'll keep making progress in same direction so we want it to travell faster in that direction. So moving its speed to move the losss funtion in that direction quickly is called as momentum.


* when parameters increase `Regularization` becomes important.