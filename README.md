## Factorisation machines


W positive def  matrix can be represented by cholesky decomposition 'for large enough k' ?
claim FM can represent any interaction matrix.

Factorisation machines represent symmetric matrices
and 

why is W +ve def?
a symmetric matrix is positive definite if strictly diagonally dominant 
but the diagonal of our interaction matrix is undefined, so we can just define the diagonals to be sum of all the off diagonal row elements.
https://nhigham.com/2021/04/08/what-is-a-diagonally-dominant-matrix/

https://ojs.aaai.org/index.php/AAAI/article/view/4340


argues pos def ass is faulty


can say W is symmetric if consider non field aware matrix


The runtime complexity of calculating the prediction function 1 is
polynomial in the number of parameters, i.e. $O(p^o)$. By factorization of the model
parameters this complexity can be reduced to $O(p\sum_{i=1}^o k_o)
i=1 ko) by rearranging terms
in equation 6 to