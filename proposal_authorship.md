# Machine Learning Engineer Nanodegree
## Capstone Proposal
Kavita Kanojiya

March 10th, 2018


## Proposal
Building a prediction engine that will identify author for a given document.


### Domain Background

Since ancient days, discoveries, patents, analyses were and are often documented. It does not matter how they are documented.

They can be documented either in the form of wall paintings or wall embedded drawings or any sort of scripts.

Older documents are often referred to continue the research even now in every fields like literature, experiments, myths, etc.

However, it is necessary for the seeker or organisation that the authenticity of these documents are met. Still, there could be lots of anonymous and non-classified documents exists. 

So, how were these documents verified? In the ancient days, these were all manually verified. There are different patterns like style of writing, language of communication, etc. of understand from where these documents came in and potentially by whom. Also, do we have any similar documents identified earlier?

This is broadly known as Stylometry. Stylometry deals with the study of linguistic style and is widely adopted across academic, literary, music and fine arts. Also, known as authorship attribution.

Also, now-a-days, the anonymous nature of online-message distribution makes identity tracing a critical problem in our society where everything is online.


### Problem Statement

In the real world, there could be still unidentified and anonymous documents that weren't recognised. Studying the patterns of writing and figuring out if similar documents were ever identified in the past. If yes, how many? Calculating the time performing these manual tasks is tedious.

However, even if documents are identified and studied, the explanation has to be written somewhere for other researches to access.

Human is prone to make errors. Error and time consumption are the factors that is the need of the requirement.

Objective to enable prediction of documents in order to identify the potential author or owner of the documents which is very tedious if we handle them manually.


### Datasets and Inputs

Human race has different number of languages and they are all used for communication. This means when a document or text message is written, people choose any of the languages to do so.

Geographical regions contribute largely to semantics of the languages and one of the factor to identify the potential origin of the documents/messages.

In a nutshell, text messages will be the only attribute required to work on to start with the objective.

Now, we have set the objective of this project to identify potential owner or author of the documents. This is a feature-based classification problem.

When we say classification, we have both features and labels within the data to explore.

#### Datasets


1. Training set

Attached is the glimpse of `train.csv` file to represent the format of the data. Each row has the text message to identify the semantics and these are already identified for the authorship.

![alt text](https://github.com/kavitakanojiya/capstone-machine-learning/blob/master/author%20detection%20statistics/training_set.png "Training Set samples")

1. `id`: Each document is unique and hence, they are identified with unique ID.

2. `text`: Each document has a text as a feature for the training model.

3. `author`: Each document has an associated author to dictate the authorship.

We will limit the data to follow only English language to explore with. This will be our features.

We shall limit the number of authors too to keep it simple. This will be our labels.

Every document is identified with an author respectively.


2. Testing set

Attached is the glimpse of `test.csv` file to represent the format of the data. Each row has the text message of authorship has to be predicted.

![alt text](https://github.com/kavitakanojiya/capstone-machine-learning/blob/master/author%20detection%20statistics/testing_set.png "Testing Set samples")

1. `id`: Each document is unique and hence, they are identified with unique ID.

2. `text`: Each document has a text as a feature for the training model.

Since we are supposed to predict the author for testing data, our target variable is `author` itself.


3. Statistics of training dataset

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/author%20detection%20statistics/statistics%20of%20training%20dataset.png "Training Set statistics")


4. Pre-defined authors list

⋅⋅⋅EAP: Edgar Allan Poe

⋅⋅⋅HPL: HP Lovecraft

⋅⋅⋅MWS: Mary Wollstonecraft Shelley


5. Author's data distribution

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/author%20detection%20statistics/EAP.png "EAP: Edgar Allan Poe")

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/author%20detection%20statistics/HPL.png "HPL: HP Lovecraft")

![alt text](https://raw.githubusercontent.com/kavitakanojiya/capstone-machine-learning/master/author%20detection%20statistics/MWS.png "MWS: Mary Wollstonecraft Shelley")


### Solution Statement

Supporting wide range of languages on this earth to address this problem is a challenging task. Every regional semantic comes with their own surprises.
For example, what and how grammar being used, how letters are written, typography of the alphabets, etc. contributes to the human race.

To address this, we have to make our training model learn every aspects of these parameters. However, cleaning up the data probably should not be done since this will defeat the purpose of documents originated from the regions.

With this project, we will limit the language of documents only to be English.


### Benchmark Model

Since texts listed in the `train.csv` will be our features, I shall have multiple algorithms trained on them to calculate the accuracy for this dataset.

I will also provide explanation as how accuracy is relevant returned from any one of the algorithms.


### Evaluation Metrics

Benchmark model will be created using the training dataset. Accuracy shall be defined too. I will use the same model to predict over the testing dataset i.e `test.csv` to study the accuracy of the resultset.

As per this problem, I will need to identify the patterns of the texts that will help me to identify the patterns adopted by the author.

Since the number of authors are limited in this problem, I can pick up few examples for each author and their documents to analyse more closely. This will be a bit if manual intervention.


### Project Design

This project deals with the texts and their formation that author adopts. In real-life, the semantics of writings dictates the area that author comes from.

If we analyse how stylometry works in this case, we will come to know how people communicate in their most comfortable way using the most common language that is easily understandable by most of the people.

Primarily, I won't be cleaning up the stopwords since they will help me to identity the construction of words that author is used to.

`tf-idf` will help me to collect the keywords relevant to the documents that will help me to build the features set.

Then, the features set available will be used to train against our model.

We shall use few training models like Multinomial Naives Bayes, Logistic Regression and Random Forest to evaluate how they perform with our problem statement. Accuracy will also be calculated by tuning the hyperparameters to refine it.

Each models will lists their own accuracy scores and predicted probablities.

These models will also be tested on our testing dataset to calculate the accuracy of the data.

In the end, we can make the predictions on the documents to which author it belongs to on the testing dataset.

We shall go through the training documents to verify if similar predictions was carried out on testing documents. This will help us to learn how they were identified and any differences will be noted too.
