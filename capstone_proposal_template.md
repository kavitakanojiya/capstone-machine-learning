# Machine Learning Engineer Nanodegree
## Capstone Project
Joe Udacity  
December 31st, 2050

## I. Definition
_(approx. 1-2 pages)_

### Project Overview
Since ancient days, discoveries, patents, analyses were and are often documented. It does not matter how they are documented.

They can be documented either in the form of wall paintings or wall embedded drawings or any sort of scripts.

Older documents are often referred to continue the research even now in every fields like literature, experiments, myths, etc.

However, it is necessary for the seeker or organisation that the authenticity of these documents are met. Still, there could be lots of anonymous and non-classified documents exists. 

So, how were these documents verified? In the ancient days, these were all manually verified. There are different patterns like style of writing, language of communication, etc. of understand from where these documents came in and potentially by whom. Also, do we have any similar documents identified earlier?

This is broadly known as Stylometry. Stylometry deals with the study of linguistic style and is widely adopted across academic, literary, music and fine arts. Also, known as authorship attribution.

Also, now-a-days, the anonymous nature of online-message distribution makes identity tracing a critical problem in our society where everything is online.

We will be dealing with the classification problem on documents, texts, etc that went unidentified.

### Problem Statement

Everytime, we always look for any signs or discoveries from the ancient times. Anything that were found can never be unwritten or undocumented.

Every findings can be communicated either through painting, embeddings, drawings or on papers.

Before believing or release to other communities, we make our best efforts to find the authenticity of documents by its origin, geographical region, language semantics, etc.

Out of 200+ languages, several writing patterns, it is a bit tedious tasks to manual identify or study these documents.

However, even if documents are identified and studied, the explanation has to be written somewhere for other researches to access.

The objective here is to enable prediction of documents in order to identify the potential author or owner of the documents which is very tedious if we handle them manually.

The limitation of this project is to build an interface that will identify English documents and authors of these documents are just 3 of them.

### Metrics

Stylometry and author attribution deals with the patterns of the writings, language semantics and language itself. We have to consider stopwords and punctuations both.

As this is feature-based classification problem, we are going to use Supervised Learning algorithm to solve this.

We will need to identify the patterns of the texts that will help me to identify the patterns adopted by the author.

Before finalising the learning algorithm, we will evaluate few of the algorithms for their performance and accuracy. Benchmark model is created using the training dataset.

Primarily, we shall focus on two things:
- Since the number of authors are limited in this problem, we are generating the wordclouds for each author. This will enable us to notice the keywords that each author emphasizes.
- How words are used, how unique words are used, punctuations being used? These will be analysed with heatmaps.


## II. Analysis
_(approx. 2-4 pages)_

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

To do this, we shall use KFold to make our model learn and cross_val_score to calculate the accuracy_score for each algorithm.

Below is the statistics of performances of the selected algorithms.

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/images/algorithm_performances.png "Model performance and statistics")

Till now, Multinomial Naive Bayes yields better accuracy and training time.


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
