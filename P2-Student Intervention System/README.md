## Training and Evaluating Models  
I choose the following three supervised learning models for this dataset  
1. Support Vector Machines  
2. Gaussian Naive Bayes  
3. Some random algorithm 

Let's go over them one by one

### Support Vector Machines
**Applications of SVM**

 Support Vector Machine (SVM) is primarily a classier method that performs classification tasks by constructing hyperplanes in a multidimensional space that separates cases of different class labels.SVM supports both regression and classification tasks and can handle multiple continuous and categorical variables. As we saw in the same project, the categorical variables are turned into boolean variables by creating some dummy columns.
 
 SVMs can be used to perform both multi-class classification and regression.In addition to performing linear classification, SVMs can efficiently perform a non-linear classification by mapping their inputs into high-dimensional feature spaces.
 
 **Advantages of SVM**
 - SVMs do a great job in formulating non-linear decesion boundaries. Other supervised learning methods like Decesion Trees and Logistic regression won't give optimised results when the data is randomly distributed and there is no clear linear separation between them.
 -  By introducing the kernel, SVMs gain flexibility in the choice of the form of the threshold separating the different types of data. We can also write our custom kernels to specify the similarity between the data. This feature of writing custom kernels makes it very versatile.
 -  The SVMs gives a good generalization performance even in case of high-dimensional data and a small set of training patterns.

**Disadvantages of SVM**

