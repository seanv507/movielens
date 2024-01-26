# Factorisation machines


W positive def  matrix can be represented by cholesky decomposition 'for large enough k' ?
claim FM can represent any interaction matrix.

Factorisation machines represent symmetric matrices
with no diagonal specified.

In particular, we should be free to make the diagonals as large as we wish.


a symmetric matrix is positive definite if the diagonal is greater than the sum of all the off diagonal elements in the row.
but the diagonal of our interaction matrix is undefined, so we can just define the diagonals to be greater than the sum of all the off diagonal row elements.
https://en.wikipedia.org/wiki/Diagonally_dominant_matrix



So there exists a positive definite matrix with the same off diagonal elements.
The question is whether this is a minimum norm solution



'a Hermitian matrix M M is positive semidefinite if and only if it is the Gram matrix of some vectors'
https://en.wikipedia.org/wiki/Gram_matrix

So any $V V^T$  is positive semidefinite

Question can we find $U^T V$ such that $M$
$y(x) = \sum_i w_i x_i + \sum_{i,j} x_i M_{i,j}x_j$

We assume we can rewrite as 

$y(x) = \sum_i w_i x_i + \sum_{i,j<i, k} x_i V_{i_k} V_{j_k} x_j$

Note that $x_i x_j = 0$ unless $i$ is user and $j$ is item.

So our design matrix is low rank...?!

also we don't therefore care about corresponding quadratic terms.  

regularisation is $\|V\|^2_2=\sum_{i,k}V_{i,k}^2$.  However, what would be more natural is $\sum_{i\in U, j \in I} |\sum_k V_{i,k} V_{j,k}|^2 $


diagonals are $\|V_{i}\|^2$,

diagonal dominance requires 
$\|V_{i}\|^2 > \sum _j \|V_{i}V_{j}\|^2$.  This 'sounds' impossible?


### experiments
Given $M$ with no diagonal specified 

## Identifiability
? can we recover data generating process
is model identifiable?


data available

amount of data is relatively small


$n_u$ users, $n_i$ items, maximum rows $n_u \times  n_i$

coefficients (1 + (n_u + n_i) + (n_u n_i))

---
is symmetric matrix meaningful?

We split $V$ into $V^u\in \mathbb R^{n_u \times n_f}$ and $V^i \in \mathbb R^{n_i \times n_f}$ 

then only coefficients we care about are interaction between single user (i)  and single item (j) which is given by dot product of $V^u_i$ and $V^i_j$


instead can also think about converting dummy categories into continuous data.

$\ln(p(\text{buy} |u_i,i_j )/(1-p)) \sim b + w_u \cdot V^u_i + w_i V^i_j  + w_{u \times i}V^u_i \times V^i_j $.

Can either use gradient descent or ALS:
optimising 
1. V^u
1. V^i
1. w 

we have $(n_u + n_i + 2 + n_f )\times n_f$ coefficients.

this is not so many coefficients.



---

https://ojs.aaai.org/index.php/AAAI/article/view/4340


argues pos def ass is faulty


can say W is symmetric if consider non field aware matrix


The runtime complexity of calculating the prediction function 1 is
polynomial in the number of parameters, i.e. $O(p^o)$. By factorization of the model
parameters this complexity can be reduced to $O(p\sum_{i=1}^o k_o)
i=1 ko)$ by rearranging terms
in equation 6 to


