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
- SVMs don't work well with large datasets because the time complexity of training them is of the order of O(N^3).From a practical point of view this is the most serious problem of the SVMs is the high algorithmic complexity and extensive memory requirements of the required quadratic programming in large-scale tasks.
- Also they don't work when the data contains a lot of noise and the classes are overalpping to each other.


**Why choose this model?**  
TO BE DONE

### Gaussian Naive Bayes  
**Applications of SVM**
- Document Classification.remains a popular (baseline) method for text categorization
- Spam filtering. the problem of judging documents as belonging to one category or the other such as spam or legitimate
- It also finds application in automatic medical diagnosis
- credit approval
- Recommender system that predicts wheteher the user will like a given resource


**Advantages of Naive Bayes**  
- They require a small amount of training data to estimate the necessary parameters.
- Naive Bayes learners and classifiers can be extremely fast compared to more sophisticated methods.The decoupling of the class conditional feature distributions means that each distribution can be independently estimated as a one dimensional distribution. This in turn helps to alleviate problems stemming from the curse of dimensionality.
- Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features/predictors) in a learning problem

**Disadvantages of Naive Bayes**  
- apparently over-simplified assumptions
- On the flip side, although naive Bayes is known as a decent classifier, it is known to be a bad estimator, so the probability outputs are not to be taken too seriously.
- Naive Bayes assumes that the affect of an attribute value on a given class is independent of the value of other attributes.


**Why choose this model**  
TO BE DONE

### Neural networks
**Applications of Neural networks**
- Neural networks can receive and process vast amounts of information at once, making them useful in image compression.
- In stock market business,many factors weigh in whether a given stock will go up or down on any given day. Since neural networks can examine a lot of information quickly and sort it all out, they can be used to predict stock prices.
- The idea of using feedforward networks to recognize handwritten characters is rather straightforward. As in most supervised training, the bitmap pattern of the handwritten character is treated as an input, with the correct letter or digit as the desired output
- Neural Networks are used experimentally to model the human cardiovascular system. Diagnosis can be achieved by building a model of the cardiovascular system of an individual and comparing it with the real time physiological measurements taken from the patient.potential harmful medical conditions can be detected at an early stage 
- It can be used as an "instant physician" trained an autoassociative memory neural network to store a large number of medical records, each of which includes information on symptoms, diagnosis, and treatment for a particular case. After training, the net can be presented with input consisting of a set of symptoms; it will then find the full stored pattern that represents the "best" diagnosis and treatment.


**Advantages of Neural networks**  
- neuralnetworks can do a good job of generalization particularly on functions in which the interactions between inputs are not too intricate, and for which the output varies smoothly with the input.
- Parallel organization permits solutions to  problems where multiple constraints must be satisfied simultaneously

**Disadvantages of Neural networks**  
- Neural networks are clearly an attribute-based representation, and do not have the expressive power of general logical representations.
- MLP requires tuning a large number of hyperparameters.The solution quality of an ANN is known to be affected by the number of layers, the number of neurons at each layer, the transfer function of each neuron, and the size of the training set.
- For non trivial problems, you generally need a very large network which can be extraordinarily time intensive to evaluate at inference time. This makes them expensive for production uses.

**Why choose this model**  
TO BE DONE




