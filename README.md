# Machine-Learning-2020-CS60050-
Run different source codes for plotting as well as regression and regularization(Lasso and Ridge)
There are comments on changing the order of the polynomial, lambda1, as well as learning rate  at respective places

1.The smallest error was found to be in polynomial of order 4. Polynomial of order 9 though accurate was found to overfit the curve.
Hence not a great choice.
Hence n=4 is most suitable for our dataset as its overfitting can be reduced by just changing the regularization parameter lambda and 
nothing else needs to be done.

3. The difference between these type of regularization is error in case of ridge regression shows more gradual behaviour as compared to lasso
regression where in ridge regression error continuously increases.I will prefer Lasso over ridge as a better fit as the magnitude of regularization
term is much less in Lasso as compared to ridge but there are problems too as we need to take the absolute value of the sum of the gradients for a 
better error estimation.