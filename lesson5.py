
# crating pytorch model
# the actual work to be done should be put in the function is 
# `forward`

class DotProduct(nn.Module):
    def forward(self, u, m) : return (u * m).sum(1)

model = DotProduct()

a = T([[1, 2], [3, 4]])
b = T([[2, 2], [10, 10]])

model(a, b)

# We do not have the contigious IDs for the users or movies 
# first thing we do is map those ids to the contigious ids
# e.g. AD34 => 1 etc.

# find out how many unique userIds are there
u_uniq = ratings.userId.unique()

# give each userId an index
user2idx = {o:i for i, o in enumerate(u_uniq)}

# replace those userIds with indexes
ratings.userId = ratings.userId.apply(lambda x: user2idx[x])


# similar for movies
m_uniq = ratings.movieId.unique()
movie2idx = {o:i for i, o in enumerate(m_uniq)}
ratings.movieId = ratings.movieId.apply(lambda x: movie2idx[x])


# whenever we need a modul where we need to specify, that here are the number of
# elements we want this model to work for, we need to have a constructor
# for the class we'll build for that model

# Following is the model for the embeddings for the users and movies.
class EmbeddingsDot(nn.Module):
    def __init__(self, n_users, n_movies):
        super().__init__()
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.u.weight.data.uniform_(0, 0.05)
        self.m.weight.data.uniform_(0, 0.05)

    # here the `.weight` param is the actual embedding matrix
    # but its not a tensor its a variable 
    # which means they support exact same operations but
    # It can also do automatic differentiation.
    # To pull the tensor out of the variable we need to call 
    # `.data` attribute of the variable 
    # if we place `_` at the end of the function then the result will be
    # generated in place

                    # categorical && # contineous vars
    def forward(self, cats, conts):
        users, movies = cats[:, 0], cats[:, 1]  # these are the minibatches
        u, m = self.u(users), self.m(movies)
        return (u*m).sum(1)

    # do not loop through the minibatch items 
    # this will not use GPU. hence do the tasks the whole minibatch at a time
    

# BIAS


# colaborative filtering with neural nets

class EmbeddingNet(nn.Module)
    def __init__(self, n_users, n_movies):
        super().__init__()
        (self.u, self.m) = [get_emb(*o) for o in [
            (n_users, n_factors), (n_moviesm, n_factors)]]
        self.lin1 = nn.Linear(n_factors*2, 10)
        self.lin2 = nn.Linear(10, 1)

    def forward(self, cats, conts):
        users, movies = cats[:, 0], cats[:, 1]
        x = F.dropout(torch.cat([self.u(users), self.m(movies)], dim=1), 0.75)
        x = F.dropout(F.relu(self.lin1(x)), 0.75)
        return F.sigmoid(self.lin2(x)) * (max_rating - min_rating + 1) + min_rating - 0.5

# only in our last layer only we ues sigmoid which is nonlinear. Linear function
# makes it hard to generalize it faster.