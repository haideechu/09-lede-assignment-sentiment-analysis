#!/usr/bin/env python
# coding: utf-8

# # Designing your own sentiment analysis tool
# 
# While there are a lot of tools that will automatically give us a sentiment of a piece of text, we learned that they don't always agree! Let's design our own to see both how these tools work internally, along with how we can test them to see how well they might perform.

# ## Training on tweets
# 
# Let's say we were going to analyze the sentiment of tweets. **If we had a list of tweets that were scored positive vs. negative, we could see which words are usually associated with positive scores and which are usually associated with negative scores.** We wouldn't need VADER or pattern or anything like that, we'd be able to _know_ we had a good dataset!
# 
# Luckily, we have **Sentiment140** - http://help.sentiment140.com/for-students - a list of 1.6 million tweets along with a score as to whether they're negative or positive. We'll use it to build our own machine learning algorithm to see separate positivity from negativity.
# 
# I'm providing **sentiment140-subset.csv** for you: a _cleaned_ subset of Sentiment140 data. It contains half a million tweets marked as positive or negative.
# 
# ### Read in our data
# 
# Read in `sentiment140-subset.csv` and take a look at it.

# In[2]:


import pandas as pd
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.max_columns", 200)

# Read in your dataset
df = pd.read_csv('sentiment140-subset.csv')


# The subset is originally 500,000 tweets, but we don't have all the time in the world! I'm going to cut it down to 3,000 instead. **Be sure you run this code, or else you might be stuck training your language models for a very long time!**

# In[3]:


# In theory we would like a sample of 3000 random tweets, which you
# can do with this code:
# df = df.sample(3000)
# the problem is I'd like to say things later about specific
# tweets, so I'm going to force us to keep the first 3000 instead
df = df[:3000]


# It isn't a very complicated dataset. `polarity` is whether it's positive or not, `text` is the text of the tweet itself.
# 
# How many rows do we have? **Make sure it's 3,000.**

# In[4]:


len(df)


# How many **positive** tweets compared to how many **negative** tweets?

# In[5]:


df.polarity.value_counts()
# It seems like 0 is negative and 1 is positive?


# ## Train our model
# 
# To build our model, we're going to use a machine learning library called [scikit-learn](https://scikit-learn.org/stable/). It's a "classical" machine learning library, which means it isn't the "this is a black-box neural network doing magic that we don't understand" kind of machine learning. We'll be able to easily look inside.
# 
# You can install it with `pip install sklearn`.
# 
# > This section is going to be a lot of cut and paste/just running code I've already put together (and maybe tweaking it a little). We'll get deeper into sklearn as we go forward in our machine learning journey!

# In[6]:


get_ipython().system('pip install sklearn')


# ### Counting words
# 
# Remember how we could just make a word cloud and call it a language model? We're going to do the same thing here! It's specifically going to be a **bag of words** model, where we don't care about the order that words are in.
# 
# It's also going to do a little trick that makes **less common words more meaningful.** This makes common words like `the` and `a` fade away in importance. Technically speaking this "little trick" is called TF-IDF (term-frequency inverse-document-frequency), but all you need to know is "the more common a word is, the less we'll pay attention to it."
# 
# The code below creates a `TfidfVectorizer` – a fancy word counter – and uses it to convert our tweets into word counts.
# 
# **Since we don't have all the time and energy in the world and want to keep our CO2 to a minimum,** let's only take a selection of words. We can use `max_features` to only take the most common words - let's try the top 1000 for now.

# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[8]:


vectorizer = TfidfVectorizer(max_features=1000)
vectors = vectorizer.fit_transform(df.text)
words_df = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names())
words_df.head()


# Each word (or token, as we learned!) gets a column, and each tweet gets a row. A zero means the word did not show up in the tweet, while any other number means it did. A score of `1.0` means it's the only word in the tweet (or the only word that the language model is paying attention to).
# 
# For example, you see `0.427465` under `10` for the fourth tweet. That means `10` was a pretty important word in the fourth tweet! In the same vein, if you scroll to the far far right you can see our first tweet got a score under `you` for `0.334095`.
# 
# Tweets aren't very long so you usually have only a handful of non-zero values for each row. If each row was a book with a lot of words, you'd have lower values spread out across all of the words.
# 
# ### Checking our word list
# 
# Use `vectorizer.get_feature_names()` to look at the words that were chosen. Do you have any thoughts or feelings about this list?

# In[9]:


vectorizer.get_feature_names()


# In[10]:


# I don't have any feelings about it because it doens't mean much to me right now.
# I don't know the context in which these words are used or how frequently they were used.


# ### Setting up our variables and training a language model
# 
# Now we'll use our word counts to build a language model that can do sentiment analysis! Because we want to fit in with all the other progammers who use machine learning, we need to create two variables: one called `X` and one called `y`.
# 
# `X` is our **features**, the things we use to predict positive or negative. In this case, it's going to be our words. We'll be using words to predict whether a tweet is positive or negative.
# 
# `y` is our **labels**, the positive or negative rating that we want to predict. We'll use the `polarity` column for that.

# In[11]:


X = words_df
y = df.polarity


# ### Picking an architecture
# 
# We talked about picking an **architecture** in class. To a large degree, a model (language model, vision model, etc) is a combination of an architecture, a dataset, and a handful of other choices. The models we talked about in class were mostly "neural nets" that had components like "bidirectional masking" and other buzzwords we couldn't understand. It's the exact same thing for classical machine learning!
# 
# So what kind of architecture do we want? Who knows, we don't know anything about machine learning! **Let's just pick ALL OF THEM.**
# 
# > **Sidenote:** Blindly picking multiple architectures and seeing which one performs the best is a completely valid thing to do in data science. To a large degree, it's a lot of "if it works, it works! who cares why?"

# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


# ### Training our language models
# 
# When we teach our language model about what a positive or a negative tweet looks like, this is called **training**. Training can take different amounts of time based on what kind of algorithm you are using.
# 
# For the scikit-learn library, you use `.fit(X, y)` to teach a model how to predict the labels (`y`: positive, negative) from the features (`X`: the word usage).

# In[13]:


get_ipython().run_cell_magic('time', '', "# Create and train a logistic regression\nlogreg = LogisticRegression(C=1e9, solver='lbfgs', max_iter=1000)\nlogreg.fit(X, y)")


# In[14]:


get_ipython().run_cell_magic('time', '', '# Create and train a random forest classifier\nforest = RandomForestClassifier(n_estimators=50)\nforest.fit(X, y)')


# In[15]:


get_ipython().run_cell_magic('time', '', '# Create and train a linear support vector classifier (LinearSVC)\nsvc = LinearSVC()\nsvc.fit(X, y)')


# In[16]:


get_ipython().run_cell_magic('time', '', '# Create and train a multinomial naive bayes classifier (MultinomialNB)\nbayes = MultinomialNB()\nbayes.fit(X, y)')


# **How long did each take to train?** Were any much faster than others? While we didn't fly any planes across the ocean to build these, at the very least a model that takes a long time to train can be *annoying*.

# In[17]:


# Seems like logreg took the most time and bayes took tthe least


# ## Use our models
# 
# Now that we've trained our language models, **we can use them to predict whether some text is positive or negative**.
# 
# ### Preparing the data
# 
# I started us off, but **add a few more sentences below.** They should be a mix of positive and negative. They can be boring, they can be exciting, they can be short, they can be long. Honestly, you could paste a book in there if you were dedicated enough.

# In[18]:


# Create some test data
unknown = pd.DataFrame({'content': [
    "I love love love love this kitten",
    "I hate hate hate hate this keyboard",
    "I'm not sure how I feel about toast",
    "Did you see the baseball game yesterday?",
    "The package was delivered late and the contents were broken",
    "Trashy television shows are some of my favorites",
    "I'm seeing a Kubrick film tomorrow, I hear not so great things about it.",
    "I find chirping birds irritating, but I know I'm not the only one",
     "My cat is throwing up hairballs constantly and I'm worried",
    "It rained when I went to the botanical garden but it was still magical and delightful.",
    "I just got bit by multiple mosquitos. WTF.",
    "I don't really care for celeries, but I do like rhubarb",
    "Did you hear that the Bucks won the NBA Championship?",
    "They forgot my food and the delivery guy was late"
]})
unknown


# First we need to **vectorize** our new sentences into numbers, so the language model can understand them. In this case, we're doing the fancy word counting we talked about before.
# 
# Our algorithm only knows **certain words.** It learned them when we were training it! Run `vectorizer.get_feature_names()` to remind yourself of the words the vectorizer knows.

# In[19]:


vectorizer.get_feature_names()


# Run the code below to complete `unknown_words_df`, the word counts for all of the texts we wrote above.
# 
# > When I say "word counts" I mean "TF-IDF word counts that are word counts but adjusted in a very specific way to make more common words less important" (but you knew that already!)
# 
# It **only counts words that were in the training data**, because those are the only words it can understand as being positive or negative. Any new or unknown words will be thrown out!

# In[20]:


# Put it through the vectorizer
unknown_vectors = vectorizer.transform(unknown.content)
unknown_words_df = pd.DataFrame(unknown_vectors.toarray(), columns=vectorizer.get_feature_names())
unknown_words_df.head()


# Notice how it only has 1,000 rows: those are the 1,000 features (words) that we told our model to pay attention to.
# 
# Now that we've counted the words for the sentences of unknown sentiment, **we can use our model to make predictions about whether they're postive or negative.**

# ### Predicting with our models
# 
# To make a prediction for each of these new, unknown-sentiment sentences, we can use `.predict` with each of our models. For example, it would look like this for logistic regression:
# 
# ```python
# unknown['pred_logreg'] = logreg.predict(unknown_words_df)
# ```
# 
# To add the prediction for the "random forest," we'd run similar `forest.predict` code, which will give you a `0` (negative) or a `1` (positive).
# 
# #### But: probabilities!
# 
# **We don't always want just a `0` or a `1`, though**. That "YES IT'S POSITIVE" or "NO, IT'S NEGATIVE" energy is very forceful but not always appropriate: sometimes a sentence is just *kind of* positive or there's just a *little bit of a chance* that it's negative, and we're interested in the *degree*.
# 
# To know the *chance* that something is positive, we can use this code:
# 
# ```python
# unknown['pred_logreg_prob'] = linreg.predict_proba(unknown_words_df)[:,1]
# ```
# 
# **Add these new columns for each of the models you trained** - `logreg`, `forest`, `svc` and `bayes`. Everything except for LinearSVC can also do `.predict_proba`, so you should add those values as columns as well.
# 
# * **Tip:** Tab is helpful for knowing whether `.predict_proba` is an option for a given model.
# * **Tip:** Don't forget the `[:,1]` after `.predict_proba`! It means "give me the probability that it's category `1` (aka positive)

# In[22]:


# Predict using all our models. 

# Logistic Regression predictions + probabilities
unknown['pred_logreg'] = logreg.predict(unknown_words_df)
unknown['pred_logreg_proba'] = logreg.predict_proba(unknown_words_df)[:,1]

# Random forest predictions + probabilities
unknown['pred_forest'] = forest.predict(unknown_words_df)
unknown['pred_forest_proba'] = forest.predict_proba(unknown_words_df)[:,1]

# SVC predictions (doesn't support probabilities)
unknown['pred_svc'] = svc.predict(unknown_words_df)

# Bayes predictions + probabilities
unknown['pred_bayes'] = bayes.predict(unknown_words_df)
unknown['pred_bayes_proba'] = bayes.predict_proba(unknown_words_df)[:,1]


# Once you're done making your predictions, **let's look at the results!**

# In[23]:


unknown


# ### Questions

# **What do the numbers mean?** What's the difference between a 0 and a 1? A 0.5? (I don't *think* you should have any negative numbers)

# In[ ]:


# The numbers represents each model's prediction of the sentence's polarity.
# It represents the probability that the sentence is positive.
# The number falls within a scale from 0 to 1: 0 is negative and 1 is positive.
# For example, if it's more than 0.5 it means it's more positive than negative.
# If it's less than 0.5 it's more negative than positive. 0.5 means it's neutral.


# **Were there any sentences where the language models seemed to disagree about?** How do you feel about the amount they disagree? Do any of the disagreements make you specific models are useless/super smart?

# In[ ]:


# Every model but Bayes predicted that "Trashy television shows are some of my favorites" is negative.
# While forest thought that "I just got bit by multiple mosquitos. WTF." was a positive sentence.
# It also thought that "My cat is throwing up hairballs constantly and I'm worried" was positive.
# The three models agree sometimes but also disagree in many instances.
# Some seem to pay more attention to adjectives than contetxt. All in all, forest seems the most prone to mistakes.


# **What's the difference between using a simple 0/1 to talk about sentiment compared to the range between 0-1?** When might you use one or the other?

# In[ ]:


# The simple 0/1 binary offers a summary determination of whether the sentence is positive or negative
# while the 0-1 range offers room for much more nuance and discretion.
# A 0/1 binary may prove helpful when working with large-scale datasets that do not involve too much linguistic subtleties
# It could also be helpful and maybe reliable when the model is built specifically for the kind of analysis you're doing.
# A 0-1 range, however, seems to offer more nuanced and accurate judgements.


# **Between 0-1, what range do you think counts as "negative," "positive" and "neutral"?** For example, are things positive as soon as you hit 0.5? Or does it take getting to 0.7 or 0.8 or 0.95 to really be able to call something "positive"?

# In[ ]:


# Negative: 0-0.3
# Somewhat negative: 0.3-0.4
# Neutral: 0.4 - 0.6
# Somewha positive: 0.6-0.7
# Positive: 0.7 - 1


# ## Testing our models
# 
# Instead of talking about our *feelings* about which model is our favorite, **we can actually test our language models to see which performs the best!** Our metrics aren't going to end up on [paperswithcode.com](https://paperswithcode.com/) but they'll be good enough for us.
# 
# Remember our original tweets, the ones we used to train our models? We were able to teach our model what a positive and a negative tweet was because each tweet was marked as positive or negative.
# 
# To see how good our model is, we can give each model a known tweet and say "is this positive or negative?" Then we'll compare the result to what's in our dataset. If the tweet was positive, did it predict positive?

# In[24]:


# Let's remind ourselves what our data looks like
df.head()


# Our original dataframe is a list of many, many tweets. We turned this into `X` - vectorized words - and `y` - whether the tweet is negative or positive.
# 
# Before we used `.fit(X, y)` to train each model on all of our data, so we have these wonderful pre-trained models now. **But if we're testing our language model on a tweet it's already seen, isn't that kind of like cheating?** It already knows the answer!
# 
# Instead, we'll give our models 80% of our tweets as training data to learn from, and then keep 20% separate to quiz it on later. It's like when a teacher gives you a study guide that's *similar* to what will be on the test, but not *exactly* the same.
# 
# This is called a **train-test split**, and you always use the exact same code to do it. Yes, the models would be smarter if we gave it all of the data, but then we wouldn't be able to test it!

# In[25]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)


# > Note about real life: When deploying a model into actual use, you typically pick the best-performing model after train/test split evaluation and then train it *again* using all of your data. If it was the best with 80% of the data it's probably even better with 100% of the data! Kind of like how you like to have homework answer keys after you turn the homework in.
# 
# Now that we've split our tweets into training and testing tweets, we can use our training data to teach our model what positive and negative tweets look like. **Add training for random forest, linear SVC, and Naive Bayes models.**
# 
# Later we'll see how accurate it is when looking at the other 20% of the tweets.

# In[26]:


print("Training logistic regression")
logreg.fit(X_train, y_train)

print("Training random forest")
forest.fit(X_train, y_train)

print("Training SVC")
svc.fit(X_train, y_train)

print("Training Naive Bayes")
bayes.fit(X_train, y_train)


# ### Confusion matrices
# 
# To see how well each model performs on the test dataset, we'll use a ["confusion matrix"](https://en.wikipedia.org/wiki/Confusion_matrix) for each one. I think confusion matrices are called that because they are confusing.
# 
# **We'll talk about them a lot more in class because they're my favorite thing on the entire planet.**

# In[28]:


from sklearn.metrics import confusion_matrix


# #### Logistic Regression confusion matrix
# 
# The basic idea of a confusion matrix is it **compares the actual values to the predicted values for each tweet.** It's just like how a teacher would compare the answers on your quiz to the answer key.
# 
# If the language model predicts the same as the actual answer, great! But instead of just giving you the percent you got correct, the benefit of a confusion matrix is that **it also tells you which types of questions you got wrong.** 
# 
# For example, we can know if we always accidentally predict negative tweets as positive ones. That's more useful than just knowing we got 75% correct!

# In[29]:


y_true = y_test
y_pred = logreg.predict(X_test)
matrix = confusion_matrix(y_true, y_pred)

label_names = pd.Series(['negative', 'positive'])
pd.DataFrame(matrix,
     columns='Predicted ' + label_names,
     index='Is ' + label_names)


# In[30]:


# Yes, we can also be lazy and ask for just the score
logreg.score(X_test, y_test)


# #### Random forest

# In[31]:


# YOUR CODE HERE
# Add a confusion matrix for the random forest
y_true = y_test
y_pred = forest.predict(X_test)
matrix = confusion_matrix(y_true, y_pred)

label_names = pd.Series(['negative', 'positive'])
pd.DataFrame(matrix,
     columns='Predicted ' + label_names,
     index='Is ' + label_names)


# In[32]:


# YOUR CODE HERE
# Find the overall score for the random forest
forest.score(X_test, y_test)


# #### SVC

# In[33]:


# YOUR CODE HERE
# Add a confusion matrix for the linear SVC
y_true = y_test
y_pred = svc.predict(X_test)
matrix = confusion_matrix(y_true, y_pred)

label_names = pd.Series(['negative', 'positive'])
pd.DataFrame(matrix,
     columns='Predicted ' + label_names,
     index='Is ' + label_names)


# In[34]:


# YOUR CODE HERE
# Find the overall score for the linear SVC
svc.score(X_test, y_test)


# #### Multinomial Naive Bayes

# In[35]:


# YOUR CODE HERE
# Add a confusion matrix for the naive bayes
y_true = y_test
y_pred = bayes.predict(X_test)
matrix = confusion_matrix(y_true, y_pred)

label_names = pd.Series(['negative', 'positive'])
pd.DataFrame(matrix,
     columns='Predicted ' + label_names,
     index='Is ' + label_names)


# In[36]:


# YOUR CODE HERE
# Find the overall score for the naive bayes
bayes.score(X_test, y_test)


# ### Percentage-based confusion matrices
# 
# Sometimes it's kind of irritating that they're just raw numbers. With a little crazy code, we can calculate them as percentages instead.

# #### Logisitic regression

# In[37]:


y_true = y_test
y_pred = logreg.predict(X_test)
matrix = confusion_matrix(y_true, y_pred)

label_names = pd.Series(['negative', 'positive'])
pd.DataFrame(matrix,
     columns='Predicted ' + label_names,
     index='Is ' + label_names).div(matrix.sum(axis=1), axis=0)


# Out of all of the negative tweets, what percent did we accurately predict?

# In[ ]:


# 65.76 percent


# Did we do better predicting negative tweets or positive tweets?

# In[ ]:


# It's better at predicting negative tweets


# #### Random forest

# In[39]:


# YOUR CODE HERE
# Calculate a percentage-based confusion matrix for the random forest
y_true = y_test
y_pred = forest.predict(X_test)
matrix = confusion_matrix(y_true, y_pred)

label_names = pd.Series(['negative', 'positive'])
pd.DataFrame(matrix,
     columns='Predicted ' + label_names,
     index='Is ' + label_names).div(matrix.sum(axis=1), axis=0)


# How does the random forest compare to the logistic regression?

# In[ ]:


# It's slightly better at predicting negative tweets and about the same as proficient at predicting positive ones.


# #### Linear SVC

# In[41]:


# YOUR CODE HERE
# Calculate a percentage-based confusion matrix for linear SVC
y_true = y_test
y_pred = svc.predict(X_test)
matrix = confusion_matrix(y_true, y_pred)

label_names = pd.Series(['negative', 'positive'])
pd.DataFrame(matrix,
     columns='Predicted ' + label_names,
     index='Is ' + label_names).div(matrix.sum(axis=1), axis=0)


# The linear SVC doesn't do as well as the random forest, but it does have one benefit. **Can you remember what it was?** We discovered it even before we used our models!

# In[ ]:


# For some reason my linear SVC did better than random forest?
# Anways, Linear SVC's benefit is that it's generally faster than other models.


# #### Multinomial Naive Bayes

# In[42]:


# YOUR CODE HERE
# Calculate a percentage for naive bayes
y_true = y_test
y_pred = bayes.predict(X_test)
matrix = confusion_matrix(y_true, y_pred)

label_names = pd.Series(['negative', 'positive'])
pd.DataFrame(matrix,
     columns='Predicted ' + label_names,
     index='Is ' + label_names).div(matrix.sum(axis=1), axis=0)


# ## Review
# 
# If you find yourself unsatisfied with a tool, you can try to build your own! This is exactly what we tried to do, using the **Sentiment140 dataset** and several machine learning algorithms.
# 
# Sentiment140 is a database of tweets that come pre-labeled with positive or negative sentiment, assigned automatically by presence of a `:)` or `:(`.  Our first step was using a **vectorizer** to convert the tweets into numbers a computer could understand.
# 
# After that, we built four different **language models** using different machine learning algorithms. Each one was fed a list of each tweet's **features** - the words - and each tweet's **label** - the sentiment - in the hopes that later it could predict labels if given a new tweets. This process of teaching the algorithm is called **training**.
# 
# In order to test our algorithms, we split our data into two parts - **train** and **test** datasets. You teach the algorithm with the first group, and then ask it for predictions on the second set. You can then compare its predictions to the right answers and view the results in a **confusion matrix**.
# 
# Although **different algorithms took different amounts of time to train**, they all ended up with over 70%+ accuracy.

# ## Discussion topics

# Which models performed the best? Were there big differences?

# In[ ]:


# Overall, Bayes performed best. There's just about 5 percent difference between the models.


# **Do you think it's more important to be sensitive to negativity or positivity?** Do we want more positive things incorrectly marked as negative, or more negative things marked as positive?
# 
# If your answer is "it depends," give me an example.

# In[ ]:


# It depends. I think it's better to be more sensitive to negatives if you're at the early stages of your reporting
# and is using the models to identify bad things that deserve to be investigated: For example, if you're trying to 
# flag sentences from complaints, inspections reports, evaluations and product reviews, etc.
# It might be better to have the models be more senstive to positives if you're curious to
# identify things that are, well, also positive. For example, if you are trying to identify tweets that come from
# bots that spread propoganda about how great something is.


# **Our models all had very different training times.** Which model(s) do you think offer the best combination of performance and not making you wait around for an hour?

# In[ ]:


# Based on just this exercise, it seems that Bayes is both the fastest and the most accurate


# In the Gebru paper, "language model size" was discussed frequently. Google, Facebook, Microsoft and others are all trying to build larger and larger models in the hopes that they do a better job representing language.
# 
# **What are two ways we could increase our model size?**

# In[ ]:


# 1. We can set a higher threshold for "max_features" when we set up our vectorizer
# 2. Or we can feed the model more sentences/tweets


# If you're feeling like having a wild time, **experiment with how increasing your model size affects training time and accuracy.** You'll just need to change a few numbers and run all of the cells again.

# In[ ]:





# Is 75% accuracy good?

# In[ ]:


# It's OK but I'm also not entirely comfortable or trusting of it.


# Do your feelings change if the performance is described as "incorrect one out of every four times?"

# In[ ]:


# It sounds slightly worse to me. 


# If you randomly guessed positive or negative for each tweet, what would (roughly) your performance be?

# In[ ]:


# Maybe 80/85 percent? Maybe I'm overestimating myself? I'm not sure.


# **How do you feel about sentiment analysis?** Did this and/or the previous notebook make you feel any differently about it?

# In[ ]:


# It can be helpful but I would be VERY careful about using it as empirical evidence for something that will be published
# It's cool but not really publishable unless you're using a model that is specifcally trained for what you're doing
# Or if you have the human resoures to vet or further research the results.


# What would you feel comfortable using our sentiment classifier for?

# In[ ]:


# It's good for identifying things to pay attention to (i.e. narrowing the scope of a research)
# Or to roughly separate things into positive/negative categories before further assessment and sorting.

