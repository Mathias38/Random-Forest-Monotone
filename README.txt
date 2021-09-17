README

Random Forest Monotone is a research code. 
A statistical model is a parametric model because the distribution is known. 
A machine learning model is non-parametric, no assumption is made on the distribution. 
We want to create semi-parametric models that integrate some a priori knowledge on the 
shape of  the data. Monotonicity can be positive or negative. Positive monotonicity 
between an explanatory variable x  and the response y means that for xi+1 > xi 
then yi+1 >= yi. To constrain the monotonicity  the drop outs method is used. 
The method consists in deleting the trees which do not respect  certain criteria.
A monotonicity test is performed on randomly generated lines for each 
variable of interest. The method allows to constrain in a soft way the 
monotonicity on one or several variables. 

HOW TO USE :

Run example code to test the algorithm. The method fit_with_monotony_dropouts has
8 parameters : 
1- attributes x data (list)
2- response y data (list)
3- number of test rows
4- monotone postivie variables' indexes (list)
5- monotone negative variables' indexes (list)
6- target monotony for each variable (list)
7- number of tests by test row for positive variables (list)
8- number of tests by test row for negative variables (list)