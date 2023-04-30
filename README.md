Download Link: https://assignmentchef.com/product/solved-fys-stk3155-project-2-classification-and-regression
<br>
The main aim of this project is to study both classification and regression problems, where we can reuse the regression algorithms studied in project 1. We will include logistic regresion for classification problems and write our own multilayer perceptron code for studying both regression and classification problems. The codes developed in project 1, including bootstrap and/or crossvalidation as well as the computation of the mean-squared error and/or the <em>R</em>2 or the accuracy score (classification problems) functions can also be utilized in the present analysis.

The data sets that we propose here are (the default sets)

<ul>

 <li>Regression (fitting a continuous function). In this part you will need to bring up your results from project 1 and compare these with what you get from you Neural Network code to be developed here. The data sets could be

  <ol>

   <li>Either the Franke function or the terrain data from project 1, or data sets your propose.</li>

  </ol></li>

 <li>Here you will also need to develop a Logistic regression code that you will use to compare with the Neural Network code. The data set we propose are

  <ol>

   <li>The credit card data set from <a href="https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients">UCI</a><a href="https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients">.</a> This data set links to a <a href="https://bradzzz.gitbooks.io/ga-seattle-dsi/content/dsi/dsi_05_classification_databases/2.1-lesson/assets/datasets/DefaultCreditCardClients_yeh_2009.pdf">recent </a><a href="https://bradzzz.gitbooks.io/ga-seattle-dsi/content/dsi/dsi_05_classification_databases/2.1-lesson/assets/datasets/DefaultCreditCardClients_yeh_2009.pdf">scientific article</a><a href="https://bradzzz.gitbooks.io/ga-seattle-dsi/content/dsi/dsi_05_classification_databases/2.1-lesson/assets/datasets/DefaultCreditCardClients_yeh_2009.pdf">.</a> Furthermore, in the lecture slides on <a href="https://compphysics.github.io/MachineLearning/doc/pub/LogReg/html/LogReg.html">Logistic Re</a><a href="https://compphysics.github.io/MachineLearning/doc/pub/LogReg/html/LogReg.html">gression</a><a href="https://compphysics.github.io/MachineLearning/doc/pub/LogReg/html/LogReg.html">,</a> you will find an example code for the Credit Card data. You could use this code as an example on how to read the data and use <strong>Scikit-Learn </strong>to run a classification problem.</li>

  </ol></li>

</ul>

<em> </em>c 1999-2019, “Data Analysis and Machine Learning

FYS-STK3155/FYS4155″:”http://www.uio.no/studier/emner/matnat/fys/FYS3155/index-

eng.html”. Released under CC Attribution-NonCommercial 4.0

license

However, if you would like to study other data sets, feel free to propose other sets. What we listed here are mere suggestions from our side. If you opt for another data set, consider using a set which has been studied in the scientific literature. This makes it easier for you to compare and analyze your results. It is also an essential elements of the scientific discussion.

In particular, when developing your own Logistic Regression code for classification problems, the so-called Wisconsin Cancer data (which is a binary problem, benign or malignant tumors) may be studied. You find more information about this at the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html">Scikit-Learn site</a> or at the <a href="https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)">University of California at Irvine</a><a href="https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)">.</a> The <a href="https://compphysics.github.io/MachineLearning/doc/pub/DimRed/html/DimRed.html">lecture slides on dimensionality reduction have several code examples on this </a><a href="https://compphysics.github.io/MachineLearning/doc/pub/DimRed/html/DimRed.html">data set</a><a href="https://compphysics.github.io/MachineLearning/doc/pub/DimRed/html/DimRed.html">.</a>

<strong>Part a): Write your Logistic Regression code, first step.            </strong>If you opt for the credit card data, your first task is to familiarize yourself with the data set and the scientific article. We recommend also that you study the code example in the <a href="https://compphysics.github.io/MachineLearning/doc/pub/LogReg/html/LogReg.html">Logistic Regression</a><a href="https://compphysics.github.io/MachineLearning/doc/pub/LogReg/html/LogReg.html">.</a>

Write the part of the code which reads in the data and sets up the relevant data sets.

<strong>Part b): Write your Logistic Regression code, second step. </strong>Now you should write your Logistic Regression code with the aim to reproduce the Logistic Regression analysis of the <a href="https://bradzzz.gitbooks.io/ga-seattle-dsi/content/dsi/dsi_05_classification_databases/2.1-lesson/assets/datasets/DefaultCreditCardClients_yeh_2009.pdf">scientific article</a><a href="https://bradzzz.gitbooks.io/ga-seattle-dsi/content/dsi/dsi_05_classification_databases/2.1-lesson/assets/datasets/DefaultCreditCardClients_yeh_2009.pdf">.</a>

Define your cost function and the design matrix before you start writing your code.

In order to find the optimal parameters of your logistic regressor you should include a gradient descent solver, as discussed in the <a href="https://compphysics.github.io/MachineLearning/doc/pub/Splines/html/Splines-bs.html">gradient descent lectures</a><a href="https://compphysics.github.io/MachineLearning/doc/pub/Splines/html/Splines-bs.html">. </a>Since we don’t have so many data points, you may just code the standard gradient descent with a given learning rate, or even attempt to use the NewtonRaphson method. Alternatively, it may be useful for the next part on neural networks to implement a stochastic gradient descent with and without minibatches. Stochastic gradient with mini-batches may give the best results. You could finally compare your code with the output from <strong>scikit-learn</strong>’s toolbox for optimization methods applied to logistic regression.

To measure the performance of our classification problem we use the so-called <em>accuracy </em>score. The accuracy is as you would expect just the number of correctly guessed targets <em>t<sub>i </sub></em>divided by the total number of targets. A perfect classifier will have an accuracy score of 1.

P<em>ni</em>=1 <em>I</em>(<em>t</em><em>i </em>= <em>y</em><em>i</em>)

Accuracy =   <em>,</em>

<em>n</em>

where <em>I </em>is the indicator function, 1 if <em>t<sub>i </sub></em>= <em>y<sub>i </sub></em>and 0 otherwise if we have a binary classifcation problem. Here <em>t<sub>i </sub></em>represents the target and <em>y<sub>i </sub></em>the outputs of your Logistic Regression code.

You can compare your own results with those obtained using <strong>scikit-learn</strong>.

As stated in the introduction, it can also be useful to study other datasets. In particular, when developing your own Logistic Regression code for classification problems, the so-called Wisconsin Cancer data (which is a binary problem, benign or malignant tumors) may be studied. You find more information about this at the <a href="https://scikit-learn.org/stable/modules/generated/sklearn/%20.datasets.load_breast_cancer.html">Scikit-Learn site</a> or at the “University of California at Irvine”:”https://archive.ics.uci.e du/ml/datasets/breast+cancer+wisconsin+(original)”. The <a href="https://compphysics.github.io/MachineLearning/doc/pub/DimRed/html/DimR/%20ed.html">lecture slides on dimensionality reduction have sev eral code examples on </a><a href="https://compphysics.github.io/MachineLearning/doc/pub/DimRed/html/DimR/%20ed.html">this data set</a><a href="https://compphysics.github.io/MachineLearning/doc/pub/DimRed/html/DimR/%20ed.html">.</a>

<strong>Part c): Writing your own Neural Network code. </strong>Your aim now, and this is the central part of this project, is to write to your own Feed Forward Neural Network code implementing the back propagation algorithm discussed in the <a href="https://compphysics.github.io/MachineLearning/doc/pub/NeuralNet/html/NeuralNet-bs.html">lecture slides</a><a href="https://compphysics.github.io/MachineLearning/doc/pub/NeuralNet/html/NeuralNet-bs.html">.</a> We start with the Logistic Regression case and the data set discussed in parts a) and b) but train now the network to find the optimal weights and biases. You are free to use the codes in the above lecture slides as starting points.

Discuss again your choice of cost function.

Train your network and compare the results with those from your Logistic Regression code. You should test your results against a similar code using <strong>ScikitLearn </strong>(see the examples in the above lecture notes) or <strong>tensorflow/keras</strong>.

Comment your results and give a critical discussion of the results obtained with the Logistic Regression code and your own Neural Network code. Make an analysis of the regularization parameters and the learning rates employed to find the optimal accurary score.

A useful reference on the back progagation algorithm is <a href="http://neuralnetworksanddeeplearning.com/">Nielsen’s book</a><a href="http://neuralnetworksanddeeplearning.com/">.</a> It is an excellent read.

<strong>Part d): Regression analysis using neural networks. </strong>Here we will change the cost function for our neural network code developed in part c) in order to perform a regression (fitting a function or some data set) analysis. As stated above, our default data sets could be either the Franke function or the terrain data from project 1.

Compare you results from the neural network regression analysis (with a discussion of learning rates and regularization parameters) with those you obtained in project 1. Alternatively, if you opt for other data sets, you would need to run your standard ordinary least squares, Ridge and Lasso calculations using your codes from project 1.

Again, we strongly recommend that you compare your own neural Network code and results against a similar code using <strong>Scikit-Learn </strong>(see the examples in the above lecture notes) or <strong>tensorflow/keras</strong>.

<strong>Part e) Critical evaluation of the various algorithms. </strong>After all these glorious calculations, you should now summarize the various algorithms and come with a critical evaluation of their pros and cons. Which algorithm works best for the regression case and which is best for the classification case. These codes can also be part of your final project 3, but now applied to other data sets.

<h1>Background literature</h1>

<ol>

 <li>The text of Michael Nielsen is highly recommended, see <a href="http://neuralnetworksanddeeplearning.com/">Nielsen’s book</a><a href="http://neuralnetworksanddeeplearning.com/">.</a> It is an excellent read.</li>

 <li>The textbook of <a href="https://www.springer.com/gp/book/9780387848570">Trevor Hastie, Robert Tibshirani, Jerome H. Friedman, </a><a href="https://www.springer.com/gp/book/9780387848570">The Elements of Statistical Learning, Springer</a><a href="https://www.springer.com/gp/book/9780387848570">,</a> chapters 3 and 7 are the most relevant ones for the analysis here.</li>

 <li><a href="https://arxiv.org/abs/1803.08823">Mehta et al, arXiv 1803.08823</a><a href="https://arxiv.org/abs/1803.08823">,</a> <em>A high-bias, low-variance introduction to Machine Learning for physicists</em>, ArXiv:1803.08823.</li>

</ol>

<h1>Introduction to numerical projects</h1>

Here follows a brief recipe and recommendation on how to write a report for each project.

<ul>

 <li>Give a short description of the nature of the problem and the eventual numerical methods you have used.</li>

 <li>Describe the algorithm you have used and/or developed. Here you may find it convenient to use pseudocoding. In many cases you can describe the algorithm in the program itself.</li>

 <li>Include the source code of your program. Comment your program properly.</li>

 <li>If possible, try to find analytic solutions, or known limits in order to test your program when developing the code.</li>

 <li>Include your results either in figure form or in a table. Remember to label your results. All tables and figures should have relevant captions and labels on the axes.</li>

 <li>Try to evaluate the reliabilty and numerical stability/precision of your results. If possible, include a qualitative and/or quantitative discussion of the numerical stability, eventual loss of precision etc.</li>

 <li>Try to give an interpretation of you results in your answers to the problems.</li>

 <li>Critique: if possible include your comments and reflections about the exercise, whether you felt you learnt something, ideas for improvements and other thoughts you’ve made when solving the exercise. We wish to keep this course at the interactive level and your comments can help us improve it.</li>

 <li>Try to establish a practice where you log your work at the computerlab. You may find such a logbook very handy at later stages in your work, especially when you don’t properly remember what a previous test version of your program did. Here you could also record the time spent on solving the exercise, various algorithms you may have tested or other topics which you feel worthy of mentioning.</li>

</ul>

<h1>Format for electronic delivery of report and programs</h1>

The preferred format for the report is a PDF file. You can also use DOC or postscript formats or as an ipython notebook file. As programming language we prefer that you choose between C/C++, Fortran2008 or Python. The following prescription should be followed when preparing the report:

<ul>

 <li>Use Devilry to hand in your projects, log in at <a href="http://devilry.ifi.uio.no/">http://devilry.ifi. </a><a href="http://devilry.ifi.uio.no/">no</a> with your normal UiO username and password and choose either ’fysstk3155’ or ’fysstk4155’. There you can load up the files within the deadline.</li>

 <li>Upload <strong>only </strong>the report file! For the source code file(s) you have developed please provide us with your link to your github domain. The report file should include all of your discussions and a list of the codes you have developed. Do not include library files which are available at the course homepage, unless you have made specific changes to them.</li>

 <li>In your git repository, please include a folder which contains selected results. These can be in the form of output from your code for a selected set of runs and input parameters.</li>

 <li>In this and all later projects, you should include tests (for example unit tests) of your code(s).</li>

 <li>Comments from us on your projects, approval or not, corrections to be made etc can be found under your Devilry domain and are only visible to you and the teachers of the course.</li>

</ul>

Finally, we encourage you to collaborate. Optimal working groups consist of 2-3 students. You can then hand in a common report.

<h1>Software and needed installations</h1>

If you have Python installed (we recommend Python3) and you feel pretty familiar with installing different packages, we recommend that you install the following Python packages via <strong>pip </strong>as

<ol>

 <li>pip install numpy scipy matplotlib ipython scikit-learn tensorflow sympy pandas pillow</li>

</ol>

For Python3, replace <strong>pip </strong>with <strong>pip3</strong>.

See below for a discussion of <strong>tensorflow </strong>and <strong>scikit-learn</strong>.

For OSX users we recommend also, after having installed Xcode, to install <strong>brew</strong>. Brew allows for a seamless installation of additional software via for example

<ol>

 <li>brew install python3</li>

</ol>

For Linux users, with its variety of distributions like for example the widely popular Ubuntu distribution you can use <strong>pip </strong>as well and simply install Python as

<ol>

 <li>sudo apt-get install python3 (or python for python2.7)</li>

</ol>

etc etc.

If you don’t want to install various Python packages with their dependencies separately, we recommend two widely used distrubutions which set up all relevant dependencies for Python, namely

<ol>

 <li><a href="https://docs.anaconda.com/">Anaconda</a> Anaconda is an open source distribution of the Python and R programming languages for large-scale data processing, predictive analytics, and scientific computing, that aims to simplify package management and deployment. Package versions are managed by the package management system <strong>conda</strong></li>

 <li><a href="https://www.enthought.com/product/canopy/">Enthought canopy</a> is a Python distribution for scientific and analytic computing distribution and analysis environment, available for free and under a commercial license.</li>

</ol>

Popular software packages written in Python for ML are

<ul>

 <li><a href="http://scikit-learn.org/stable/">Scikit-learn</a><a href="http://scikit-learn.org/stable/">,</a></li>

 <li><a href="https://www.tensorflow.org/">Tensorflow</a><a href="https://www.tensorflow.org/">, </a> <a href="http://pytorch.org/">PyTorch</a> and</li>

 <li><a href="https://keras.io/">Keras</a><a href="https://keras.io/">.</a></li>

</ul>

These are all freely available at their respective GitHub sites. They encompass communities of developers in the thousands or more. And the number of code developers and contributors keeps increasing.

<ol>

 <li>For project 3, you should feel free to use your own codes from projects 1 and 2, eventually write your own for SVMs and/or Decision trees and random forests’ or use the available functionality of <strong>scikit-learn</strong>, <strong>tensorflow</strong>, etc.</li>

 <li>The estimates you used and tested in projects 1 and 2 should also be included, that is the <em>R</em>2-score, <strong>MSE</strong>, cross-validation and/or bootstrap if these are relevant.</li>

 <li>If possible, you should link the data sets with exisiting research and analyses thereof. Scientific articles which have used Machine Learning algorithms to analyze the data are highly welcome. Perhaps you can improve previous analyses and even publish a new article?</li>

 <li>A critical assessment of the methods with ditto perspectives and recommendations is also something you need to include.</li>

</ol>