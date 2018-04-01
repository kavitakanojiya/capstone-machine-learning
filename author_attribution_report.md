# Machine Learning Engineer Nanodegree
## Capstone Project
Kavit Kanojiya
March 29th, 2018

## I. Definition

### Project Overview
Since ancient days, discoveries, patents, analyses were and are often documented. It does not matter how they are documented.

They can be documented either in the form of wall paintings or wall embedded drawings or any sort of scripts.

Older documents are often referred to continue the research even now in every fields like literature, experiments, myths, etc.

However, it is necessary for the seeker or organisation that the authenticity of these documents are met. Still, there could be lots of anonymous and non-classified documents exists. 

So, how were these documents verified? In the ancient days, these were all manually verified. There are different patterns like style of writing, language of communication, etc. of understand from where these documents came in and potentially by whom. Also, do we have any similar documents identified earlier?

This is broadly known as Stylometry. Stylometry deals with the study of linguistic style and is widely adopted across academic, literary, music and fine arts. Also, known as authorship attribution.

Also, now-a-days, the anonymous nature of online-message distribution makes identity tracing a critical problem in our society where everything is online.

We will be dealing with the classification problem on documents, texts, etc that went unidentified.

#### Authorship Attribution

Authorship attribution is supported by computational and statical analysis that has been around since 19th century. Due to the expansion on the communication technologies and growing population, researches, there could be and will be disputed documents about their originality and discoveries. With the growing time, text mining, information retrieval and natural language processing came into the light to solve this problems.

Researchers have been working intensely to address this issue in order to grow the community. There are bunch of papers that highlights this problem into the depth.

- [Đlker Nadi Bozkurt, Özgür Bağlıoğlu and Erkan Uyar](https://pdfs.semanticscholar.org/e629/2ad3324a248774657cb6ea0342ec50b41c7b.pdf)
- [Efstathios Stamatatos](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.440.1634&rep=rep1&type=pdf)
- [Moshe Koppel, Jonathan Schler and Shlomo Argamon](https://brooklynworks.brooklaw.edu/cgi/viewcontent.cgi?referer=https://www.google.co.in/&httpsredir=1&article=1045&context=jlp)

Most of these papers focuses on text representation, text classification and computational methods. However, authorship attribution is one of the expected outcomes from the given dataset. Moreover, it hints towards how statements are constructed, what words are emphasized, what mistakes are detected that contributes the differentiation, etc. This is all about information retrieval and text mining.

In simple words, this is very similar to what a detective do. He goes to a house under a crime scene, notices each and every objects in a house, grabs the potential hints and analyses them, talks to the suspects, understands the patterns, etc. So, relating this with the machine learning steps involved in this problem comes to be as:

Detective actions | Machine Learning
------- | ----------------
Goes into a house under a crime scene  | Read the dataset
Notices each and every objects in a house  | Analyse what dataset has (columns and rows)
Grabs the potential hints | Data visualization
Talking to the suspects | Data preprocessing
Understanding the patterns | Data exploration


### Problem Statement

Everytime, we always look for any signs or discoveries from the ancient times. Anything that were found can never be unwritten or undocumented.

Every findings can be communicated either through painting, embeddings, drawings or on papers.

Before believing or release to other communities, we make our best efforts to find the authenticity of documents by its origin, geographical region, language semantics, etc.

Out of 200+ languages, several writing patterns, it is a bit tedious tasks to manual identify or study these documents.

However, even if documents are identified and studied, the explanation has to be written somewhere for other researches to access. There are various challenges involved in this problem. Lets discuss below:

**Stylometric features**
This involves features involved in text representation for the patterns an author adopts. This includes lexical and characters features.

_Lexical features_ empahizes on the way we tokenize, stem and lemmatize the texts involved. This is a scalable option since this is not binded to any specific language.

_Character features_ empahizes on how each is represented in the texts. This involves punctuations, alphabets count, uppercased and downcased words and word frequencies.

_Semantic features_ mostly deals with NLP where we focus on part of speech that looks for grammatical construction.

However, we shall explore _Character features_ implementation. We shall consider every characters i.e punctuations, stopwords, etc to understand the patterns in the writings of each author.

The objective here is to enable prediction of documents in order to identify the potential author or owner of the documents which is very tedious if we handle them manually.

The limitation of this project is to build an interface that will identify English documents and authors of these documents are just 3 of them.

### Metrics

Stylometry and authorship attribution deals with the patterns of the writings, language semantics and language itself. We have to consider stopwords and punctuations both.

As we discussed, there are various features involved to look at this problem with different angles. They are _lexical features_, _character features_ and _semantic features_.

Considering features, _heatmap_ is one of visualizations that will help us to determine the distribution of words, what punctuations are used, how many uppercased letters are used by each author, mean word length and total length of each text.

Reflects number of words distribution of each author. X-axis grabs the run from minimum to maximum count of total number of words and Y-axis reflects the authors.

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/number_of_words.png "Number of words")


Reflects number of punctuations of each author. X-axis grabs the run from minimum to maximum count of total number of punctuations and Y-axis reflects the authors.
![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/number_of_punctuations.png "Number of punctuations")

Reflects number of unique words of each author. X-axis grabs the run from minimum to maximum count of total number of unique words and Y-axis reflects the authors.
![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/number_of_unique_words.png "Number of unique words")

These heatmaps shows the patterns that each author adopted.

Till here, we didn't get much information of author and their area of interesets. This includes the choices of words that these authors are famous for.

Wordcloud is the another visualization that helps us to analyse what words does these authors emphasizes.

Words like death, soul, life, corpse, spirit and shadow better describes his _Edgar Allan Poe_ interests. He is famous for mystery and macabre.
![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/EAP_wordclouds.png "EAP wordclouds")

Words like fear, horror and body describes his interests. So, _Howard Phillips Lovecraft_ is famous for horror.
![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/HPL_wordclouds.png "HPL wordclouds")

Words like love, affection, happiness, misery, despair, beauty and pleasure best describes _Mary Wollstonecraft Shelley_ interests.
![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/MWS_wordclouds.png "MWS wordclouds")

In all, _heatmaps_ and _wordclouds_ helped us to identify the distribution of words and what words are being used by each author.

## II. Analysis

### Data Exploration

Since the day human has evolved, their communication skills has also evolved. Geographically, their documentation will differ by the pattern they adopted, grammar they use to communicate easily, etc. This means when a document or text message is written, people choose any of the languages to do so.

Innovations and discoveries are never binded to any specific language if they are communicated well. This contributes to the semantics of the writing largely.

In essence, their writings are the only factors that needs to be discovered. Now, we have set the objective of this project to identify potential owner or author of the documents. This is a feature-based classification problem.

When we say classification, we have both features and labels within the data to explore.

For now, we shall focus on limited data to deeply dive into the problem statement.

**Training Set**

Attached is the glimpse of `train.csv` file to represent the format of the data. Each row has the text message to identify the semantics and these are already identified for the authorship.

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/training_set.png "Training Set samples")

* *_id_*: Each document is unique and hence, they are identified with unique ID.

* *_text_*: Each document has a text as a feature for the training model.

* *_author_*: Each document has an associated author to dictate the authorship.

We will limit the data to follow only English language to explore with. This will be our features.

We shall limit the number of authors too to keep it simple. This will be our labels.

Every document is identified with an author respectively.

**Testing Set**

Attached is the glimpse of `test.csv` file to represent the format of the data. Each row has the text message of authorship has to be predicted.

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/testing_set.png "Testing Set samples")

* *_id_*: Each document is unique and hence, they are identified with unique ID.

* *_text_*: Each document has a text as a feature for the training model.

Since we are supposed to predict the author for testing data, our target variable is `author` itself.

**Statistics of training dataset**

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/training_dataset_statistics.png "Training Set statistics")

**Pre-defined authors list**

* EAP: Edgar Allan Poe

* HPL: HP Lovecraft

* MWS: Mary Wollstonecraft Shelley


**Author's data distribution**

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/EAP.png "EAP: Edgar Allan Poe")

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/HPL.png "HPL: HP Lovecraft")

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/MWS.png "MWS: Mary Wollstonecraft Shelley")


### Exploratory Visualization

In this problem, we need to study how each author make use of words, punctuations, what words do they emphasize as per their interests.

We can study the distribution of semantics of each authors. This distribution includes:

1. Number of words per document.
2. Number of unique words per document.
3. Number of punctuations per document.

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/number_of_words.png "Number of words in the original document")

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/number_of_unique_words.png "Number of unique words in the original document")

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/number_of_punctuations.png "Number of punctuations in the original document")

Each author has their area of interests. For example:
* Edgar Allan Poe is best known for mystery and macabre. Words like death, soul, life, corpse, spirit and shadow better describes his interests.
* Howard Phillips Lovecraft is best known for horror fiction. Words like fear, horror and body describes his interests.
* Mary Wollstonecraft Shelley was a story writer, dramatist, essayist, biographer and traveler. Words, love, affection, happiness, misery, despair, beauty and pleasure best describes her interests.

This can be best depicted using wordclouds. Lets dive into the wordclouds for each author and what words they emphasize on.

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/EAP_wordclouds.png "Wordcloud of Edgar Allan Poe")

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/HPL_wordclouds.png "Wordcloud of Howard Phillips Lovecraft")

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/MWS_wordclouds.png "Wordcloud of Mary Wollstonecraft Shelley")

### Algorithms and Techniques

We will consider some of the supervised learning algorithms. Lets disccus about them in details one-by-one.
1. Multinomial Naive Bayes
   This algorithm remains popular due to its simplicity, scalable and wide usage in text classification problems. This works with the concept of likelihood and probabilities. This works well with the categorical input variable which is we can use it with text classification.
2. Logistic Regression
   Logistic Regression is really fast enough. Logistic regression is intrinsically simple, it has low variance and so is less prone to over-fitting. It is used in Geographic Image Processing, handwriting recognition, prediction whether a person is depressed or not based on bag of words from the corpus seems to be conveniently solvable using logistic regression and SVM, etc.
3. XGBoost
   XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.
4. SVM
   SVM works well with the problems like face detection and hand-writing detection. Effective in high dimensional space. When number of features are greater than number of samples, this will give poor performance (when training time is important). SVM is prone to overfitting and degrades when dataset has lots of noise.
5. Stochastic Gradient Descent Classifier
   Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to discriminative learning of linear classifiers such as (linear) Support Vector Machines and Logistic Regression. SGD has been successfully applied to large-scale and sparse machine learning problems often encountered in text classification and natural language processing. Efficiency and ease of implementation contributes to its advantages. However, requires a number of hyperparameters such as the regularization parameter and the number of iterations and is sensitive to feature scaling.

However, we shall simply implement with no parameters just to analyse how they perform with our dataset.

### Benchmark

We will evaluate multiple model that works well with our dataset. For this to perform, we shall make use of our training set to train our model.

Out of the training dataset, we shall pick up 80% of data that will act as our training set and the rest 20% will act as testing dataset. This will be performed repeatedly till the stipulated iterations.

This approach has an advantage that our model gets to learn on different distribution of data and does not remain bias.

To do this, we shall use KFold to make our model learn and cross_val_score to calculate the accuracy_score for each algorithm. This appears to be as:

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/kfold.png "KFold on Multinomial NB")

Below is the statistics of performances of the selected algorithms.

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/algorithm_performances.png "Model performance and statistics")

Till now, Multinomial Naive Bayes yields better accuracy and training time.


## III. Methodology

### Data Preprocessing

Since the pattern of writings is the key feature to predict the potential author for a given document, we should not be filtering out punctuations, stopwords, etc.]
This pattern helps us identify the origin of document geographically or does it belongs to any race. So, we shall perform following preprocessing steps for our learning algorithm to work with numerical data.

**Target variable `author`**
This must be numerical in order for the learning algorithm to work.
Since we have only 3 authors in the dataset, we shall label them numerically as:
* EAP: 0
* HPL: 1
* MWS: 2

And save them in a new column as `numerical_author`. This will be our target variable till we get the prediction out of the test dataset.
Once this is done, we shall remap the numerical_author to appropriate `author`.

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/processed_numerical_author.png "Numerical author")

**Features drawn from `text`**
Indeed, the text is our single feature in the dataset. However, we cannot feed to our learning algorithm. We need to vectorize it and form a new dataset altogether.

We will use CountVectorizer to create our vectors and this is our new features on which learning model will be trained.

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/processed_text_vectors.png "Processed vectors from texts")


### Implementation

In order to implement the solution for a given problem, it is utmost necessary that we understand the problem, the area in which this solution will be helpful and the background domain.
Lets go through one-by-one step in detail:

**Undergoing through domain**
Since ancient days, discoveries, patents, analyses were and are often documented. It does not matter how they are documented.

They can be documented either in the form of wall paintings or wall embedded drawings or any sort of scripts.

Older documents are often referred to continue the research even now in every fields like literature, experiments, myths, etc.

However, it is necessary for the seeker or organisation that the authenticity of these documents are met. Still, there could be lots of anonymous and non-classified documents exists.

So, how were these documents verified? In the ancient days, these were all manually verified. There are different patterns like style of writing, language of communication, etc. of understand from where these documents came in and potentially by whom. Also, do we have any similar documents identified earlier?

This is broadly known as Stylometry. Stylometry deals with the study of linguistic style and is widely adopted across academic, literary, music and fine arts. Also, known as authorship attribution.

Also, now-a-days, the anonymous nature of online-message distribution makes identity tracing a critical problem in our society where everything is online

**Datasets**
We shall hold ourselves and limit our practive with the minimum dataset. The attached dataset holds for only three authors and they are:
* _Edgar Allan Poe [EAP]_ is best known for his poetry and short stories, particularly his tales of mystery and the macabre.

* _Howard Phillips Lovecraft [HWS]_ was an American writer who achieved posthumous fame through his influential works of horror fiction.

* _Mary Wollstonecraft Shelley [MWS]_ was an English novelist, short story writer, dramatist, essayist, biographer, and travel writer.


**Data Exploration**
Data fields in the training dataset reflects following columns:
* `id` - a unique identifier for each sentence

* `text` - some text written by one of the authors and is our feature.

* `author` - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley). Hence, this is our target variable.

The data fields in testing datasets reflects follwing columns:

* `id` - a unique identifier for each sentence

* `text` - some text written by one of the authors and is our feature.

Definitely, the objective here is predict the potential author of unlabeled dataset.

We have to calculate the distribution of data that this labeled dataset provides.

However, predicting the kinds of semantics that these authors adopts, the text length, the punctuations they use, the unique words they emphaize are one of the important factors to learn the patterns in the data. Heatmap remains our choice.

Heatmap reflects X-axis with the minimum and maximum count for each pattern and Y-axis with all the 3 authors.

This is character-feature and semantic-feature implementation that consider everything in the text document.

Apart from this, these authors has their area of interests so we need to emphasize what words they choose. This is helpful to implement the lexical-feature.
Wordcloud is our choice to get the glimpse of the words that these concentrates.

**Data preprocessing**

Typically, learning algorithms expect input to be numeric, which requires that non-numeric features (called categorical variables) be converted.

Our target variable `author` in the labeled dataset is converted to numerical entries. Set records with "EAP" to 0, records with "HPL" to 1 and records with "MWS" to 2 and storing it in numerical_author column.

Converting only `author` should not only be the goal, we should perform one-hot encoding like option to our feature set `text`. CountVectorizer is an algorithm that will help to convert words in texts to be features horizontally. How the words are valid for every row in the dataset is marked as 1 and 0.

**Shuffle and split data**

`text` is our feature set and `numerical_author` is our target variable.

Machine Learning models performs better when it receives good distribution of data. For every data that belongs to a category must be trained and tested on the dataset so that model do not generate biased results.

So, from where do we get this labeled data to train and test our models.

It's simple! Our training dataset itself. Our training dataset has all the data that are well classified to each of the authors.

We shall split training dataset into 2 pieces .i.e 80% of training dataset and 20% testing dataset and this is stratified splitting as this will prevent model to generate any biased results.

**Model Evaluation and Performance**
This is what I would prefer. Pick up bunch of algorithms that would hold for this problem, run the training on the splitted dataset and calculate accuracy score. Training time is also an important factor. Of course, we do not want lon running time for a small dataset.

I, simply provided the splitted dataset to each model without any parameter tuning just to notice and learn how a model performs.

At this stage, I became pretty sure I should be using Multinomial Naive Bayes since the training time and accuracy came to outstanding here. NB is pretty well recognised when it comes to text classification. It is scalable by features wise since a text could have number of words and could be huge.

**Parameter tuning**

Every learning algorith comes with certain parameters that will work best with the dataset provided. Fine tuning them will definitely helps to increase the accuracy.
Here, `alpha` is the parameter that is required to be tuned. This definitely gave a boost to the accuracy from 0.83 to 0.93 i.e during model evaluation to parameter tuning.

**Model training**
Fitting the training dataset to the model helps the model to visit all the features and labels provided from the start.

We shall use the same 80-20 distributed data for the training.
Again, this dataset is vectorized numeric dataset from the splitted dataset.

Fit the `training_vectorizer` to the model. And run the prediction on the remaining 20% dataset.

However, this does not ends here. However, fitting and prediction is done on the labeled dataset. We can verify the values that we got from prediction and what is it in the actual dataset.

_Note:_ Fitting and Prediction gives the probability for each texts in the dataset against each author. The higher the probability is the potential author.

`Accuracy score` on the test set will yield the accuracy how a model performed on the dataset.

`classification_report` helps to learn and analyse the precision and recall scores that model performed on the dataset.

**Prediction**

Till now, we got a good control over the dataset and prediction we got. We are ready to predict the unlabeled dataset.

However, how do I verify if the prediction has got correct results?

It simple!

Wordcloud for each authors from the predicted results. Here, I have compared them with the actual dataset we processed in the early of the implementation.


### Refinement

Earlier, we chose few learning models to evaluate the performance and accuracy score. 
MNB reported accuracy of 0.834 without any parameter tuning.

In order to increase the accuracy for better results, we shall pick up the parameters that MNB supports and figure out what value works best for the dataset.
Here, the parameter used is `alpha`.

We have made use of GridSearchCV algorithm. By fitting the whole training data with alpha parameter with few values, we are able to get the best estimator that will fit well in our learning model.

Here, the best value is 0.5. Here is the quick glimpse on how it is being used:

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/parameter_tuning.png "Parameter tuning")

With this accuracy is now improved to 0.93. Here is the glimpse of the improved accuracy score on the labelled dataset.

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/improved_accuracy.png "Improved accuracy score")

## IV. Results

### Model Evaluation and Validation

This problemn is a classic problem for machine learning. We have documents that are already classified/labeled in the existing categories (here, it is authors), the task is to categorise new documents into one of the existing categories.

Naive Bayes woeks well with text classification since this follows the principle of likelihood and probability. This means we can emphasize on words that best describes a category (i.e categorical input variable or a category) making it highly scalable.

Lets discuss the approach one-byone:

**Shuffle and split data**
Machine Learning models performs better when it receives good distribution of data. For every data that belongs to a category must be trained and tested on the dataset so that model do not generate biased results.

So, from where do we get this labeled data to train and test our models.

It's simple! Our training dataset itself.

Our training dataset has all the data that are well classified to each of the authors. The only challenge is how does our model process them to create unbiased results?

Again, we shall split training dataset into 2 pieces .i.e 80% of training dataset and 20% testing dataset. Will this help?
What if model gets only sees data from authors EAP and HPL and testing set is full of MWS. Here, our model will perform poorly since it will fail to categorise he unlabeled data. Because it has only seen EAP and HPL and it will try to label them with EAP and HPL and not MWS. This is a-miss.

So, how do we ensure our model ensure that he sees all kind of data?

Again. it's simple. We already splitting the data into training and testing datasets. We shall do i recursively for iterations that we configured. Here, in each iteration, distribution of data for each author will be captured and model will start learning effectively.

Here is the short logic:
![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/shuffle_split.png "Shuffle and split data")

**Model evaluation**

We shall come up with few learning models to learn how they perform on our problem. We will use the training vector from above #1 itself.
What are those learning models?

Below are the list of models with their accuracy score and training time on ur dataset that we obtained from #1.

- Multinomial Naive Bayes
![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/multinomial_naive_bayes.png "Multinomial Naive Bayes")

- Logistic Regression
![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/logistic_regression.png "Logistic Regression")

- XGBoost from the ensemble methods
![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/xgboost.png "XGBoost")

- SVM
![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/svm.png "SVM")

- Stochastic Gradient Descent Classifier (SGDC)
![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/sgdc.png "SGDC")


[Here is the sklearn example](http://scikit-learn.org/0.15/auto_examples/grid_search_digits.html)
Also, this was also from Finding Donors project of Basic Machine Learning program.

Multinomial Naive Bayes yielded the highest accuracy with training time. We choose Multinomial NB.

**Parameter tuning**
Since the accuracy was recorded as 0.823 with no parameter. However, we shall experiment what parameters does MNB supports.

[MultinomialNB is used as the reference.](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)

GridSearchCV is used to get the best alpha value in our case. This implements the fit and predict methods and calculates what value gives the best result of accuracy.
[GridSearchCV is used as the reference.](http://scikit-learn.org/0.15/modules/generated/sklearn.grid_search.GridSearchCV.html)

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/parameter_tuning.png "Parameter tuning")

We, then, applied the best estimator to our model and found improved accuracy in the output. Previously, it reported to be 0.83 with no parameter and when parameter is tuned, we get accuracy of 0.92.

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/improved_accuracy.png "Improved accuracy")

Now, we can move ahead with this model since we see that this model works with our dataset and has improved with the parameters.

### Justification

`train.csv` is our training dataset that we vectorized, trained and predicted too if it gives the correct result since they are already labeled.
`test.csv` is our testing dataset that we need to classify among authors that we have. This is also done with the same approach i.e. vectorized, trained and predicted.

In the beginning, we analysed what are the area of interests of each author and what keywords they do emphasize. Lets see it again:

_Edgar Allan Poe_ is best known for his poetry and short stories, particularly his tales of mystery and the macabre. So, words like death, soul, life, corpse, spirit and shadow better describes his interests.

_Howard Phillips_ Lovecraft was an American writer who achieved posthumous fame through his influential works of horror fiction. So, words like fear, horror and body describes his interests.

_Mary Wollstonecraft Shelley_ was an English novelist, short story writer, dramatist, essayist, biographer, and travel writer, best known for her Gothic novel Frankenstein: or, The Modern Prometheus. So, words, love, affection, happiness, misery, despair, beauty and pleasure best describes her interests.

In the end, when we train our unlabeled dataset, we found the same keywords from text documents were properly classified among the authors.

## V. Conclusion

### Free-Form Visualization

We generated wordclouds again for all the 3 authors when the testing dataset was predicted.

Below is the wordcloud from author EAP. This focuses on keywords like soul, death and spirit as we expected from the training dataset.
![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/EAP_from_prediction.png "EAP from prediction")

Below is the wordcloud from author HPL. This focuses on keywords like horror, fear and dark as we expected from the training dataset. Also, keywords like hideous, fear and shadow were also identified.
![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/HPL_from_prediction.png "HPL from prediction")

Below is the wordcloud from author MWS. This focuses on keywords like friend, pleasure, beauty and happiness as we expected from the training dataset.
![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/MWS_from_prediction.png "MWS from prediction")

### Reflection

The most challenging task involved here are what problem it would solve and where it applicable.

Cracking any problem is not a big deal anyway now-a-days. Instead, what is the cause of the problem and where does it occurs also matters.

It required ample time to invest in researching the documents, papers that focuses on this problem, ways to solve them, what problems we should address, etc. Making decision enough as what problem to address is done by referring lots of documents and papers that were released.

Fitting, training and predicting remains global across all algorithms. Following are the challenges I would to list them down:

_Where does this problem fits in?_

This helped me explore the Stylometry and Authorship Attribution concepts. I came to know that this is not the new problem. This has been around in the existence since 10 years.

_How do I find patterns in the data?_

This helped me to identify the patterns in the writing of the given 3 authors with heatmaps. Wordclouds helped me to identify the words that these 3 authors chooses.

_How do I validate the prediction over the unlabeled dataset?_

This is something took ample time to figure out how to validate the outcomes. Nothing came out when I researched. But, instead I made use of wordclouds for the predicted unlabeled dataset and compared them with the wordclouds of the labeled dataset. 

Moreover, this comparison gave me the correct outcomes that I intend to figure out.

### Improvement
Right now, we have considered all the text documents without excluding stopwords and punctations which contributed to the increasing number of features.
But keeping in mind the that authors EAP, HPL and MWS would emphasize on certain set of keywords.

Extracting them as features and predicting would have given different results altogether. I believe predictions and classification would have been definitely improved.
