#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis Tools
# 
# Lots of libraries exist that will do sentiment analysis for you. Imagine that: just taking a sentence, throwing it into a library, and geting back a score! How convenient!
# 
# It also might be **totally irresponsible** unless you know how the sentiment analyzer was built. In this section we're going to see how sentiment analysis is done with a few different packages.

# ## Installation
# 
# Use `pip install` two language processing packages, NLTK and Textblob.

# In[2]:


# !pip install nltk textblob


# ## Tools
# 
# ### NLTK: Natural Language Tooklit
# 
# [Natural Language Toolkit](https://www.nltk.org/) is the basis for a lot of text analysis done in Python. It's old and terrible and slow, but it's just been used for so long and does so many things that it's generally the default when people get into text analysis. The new kid on the block is [spaCy](https://spacy.io/), but it doesn't do sentiment analysis out of the box so we're leaving it out of this right now.
# 
# When you first run NLTK, you need to download some datasets to make sure it will be able to do everything you want.

# In[3]:


import nltk

nltk.download('vader_lexicon')
nltk.download('movie_reviews')
nltk.download('punkt')


# To do sentiment analysis with NLTK, it only takes a couple lines of code. To determine sentiment, it's using a tool called **VADER**.

# In[4]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
sia.polarity_scores("This restaurant was great, but I'm not sure if I'll go there again.")


# Asking `SentimentIntensityAnalyzer` for the `polarity_score` gave us four values in a dictionary:
# 
# - **negative:** the negative sentiment in a sentence
# - **neutral:** the neutral sentiment in a sentence
# - **positive:** the postivie sentiment in the sentence
# - **compound:** the aggregated sentiment. 
#     
# Seems simple enough!
# 
# ### Use NLTK/VADER to determine the sentiment of the following sentences:
# 
# * I just got a call from my boss - does he realise it's Saturday?
# * I just got a call from my boss - does he realise it's Saturday? :)
# * I just got a call from my boss - does he realise it's Saturday? 😊
# 
# Do the results seem reasonable? What does VADER do with emoji and emoticons?

# In[9]:


sia.polarity_scores("I just got a call from my boss - does he realise it's Saturday?")


# In[6]:


sia.polarity_scores("I just got a call from my boss - does he realise it's Saturday? :)")


# In[7]:


sia.polarity_scores("I just got a call from my boss - does he realise it's Saturday? 😊")


# Why do you think it doesn't understand the emoji the same way it understood the emoticon?

# In[ ]:


# Maybe because emoji isn't a natural language? Maybe emojis are encoded differently?


# #### When VADER was a baby
# 
# As we talked about in class, knowing the dataset a language model was trained on can be pretty important!
# 
# [Can you uncover how VADER was trained by reading its homepage?](https://github.com/cjhutto/vaderSentiment)

# In[ ]:


# Based on the paper the homepage mentioned, it seems that VADER was trained on a dataset of social media posts.
# The homepage also mentions that its sentiment lexicon was "empirically validated by multiple independent human judges."


# ### TextBlob
# 
# TextBlob is built on top of NLTK, but is infinitely easier to use. It's still slow, but _it's so so so easy to use_. 
# 
# You can just feed TextBlob your sentence, then ask for a `.sentiment`!

# In[10]:


from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer


# In[11]:


blob = TextBlob("This restaurant was great, but I'm not sure if I'll go there again.")
blob.sentiment


# **How could it possibly be easier than that?!?!?** This time we get a `polarity` and a `subjectivity` instead of all of those different scores, but it's basically the same idea.
# 
# Try the TextBlob sentiment tool with another sentence of your own.

# In[21]:


blurb = TextBlob("This homework is kind of fun!")
blurb.sentiment


# If you like options: it turns out TextBlob actually has multiple sentiment analysis tools! How fun! We can plug in a different analyzer to get a different result.

# In[23]:


blobber = Blobber(analyzer=NaiveBayesAnalyzer())

blob = blobber("This restaurant was great, but I'm not sure if I'll go there again.")
blob.sentiment


# Wow, that's a **very different result.** To understand why it's so different, we need to talk about where these sentiment numbers come from. You can read about [the library behind TextBlob's opinions about sentiment](https://github.com/clips/pattern/wiki/pattern-en#sentiment) but they don't really go into (easily-accessible) detail about how it happens.
# 
# But first: try it with one of your own sentences!

# In[24]:


blurb = blobber("This homework is kind of fun!")
blurb.sentiment


# ## How were they made?
# 
# The most important thing to understand is **sentiment is always just an opinion.** In this case it's an opinion, yes, but specifically **the opinion of a machine.**
# 
# ### VADER
# 
# NLTK's Sentiment Intensity Analyzer works is using something called **VADER**, which is a list of words that have a sentiment associated with each of them.
# 
# |Word|Sentiment rating|
# |---|---|
# |tragedy|-3.4|
# |rejoiced|2.0|
# |disaster|-3.1|
# |great|3.1|
# 
# If you have more positives, the sentence is more positive. If you have more negatives, it's more negative. It can also take into account things like capitalization - you can read more about the classifier [here](http://t-redactyl.io/blog/2017/04/using-vader-to-handle-sentiment-analysis-with-social-media-text.html), or the actual paper it came out of [here](http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf).
# 
# **How do they know what's positive/negative?** They came up with a very big list of words, then asked people on the internet and paid them one cent for each word they scored.
# 
# ### TextBlob's `.sentiment`
# 
# TextBlob's sentiment analysis is based on a separate library called [pattern](https://www.clips.uantwerpen.be/pattern).
# 
# > The sentiment analysis lexicon bundled in Pattern focuses on adjectives. It contains adjectives that occur frequently in customer reviews, hand-tagged with values for polarity and subjectivity.
# 
# Same kind of thing as NLTK's VADER, but it specifically looks at words from customer reviews.
# 
# **How do they know what's positive/negative?** They look at (mostly) adjectives that occur in customer reviews and hand-tag them.
# 
# ### TextBlob's `.sentiment` + NaiveBayesAnalyzer
# 
# TextBlob's other option uses a `NaiveBayesAnalyzer`, which is a machine learning technique. When you use this option with TextBlob, the sentiment is coming from "an NLTK classifier trained on a movie reviews corpus."
# 
# **How do they know what's positive/negative?** Looked at movie reviews and scores using machine learning, the computer _automatically learned_ what words are associated with a positive or negative rating.
# 
# ## What's this mean for me?
# 
# When you're doing sentiment analysis with tools like this, you should have a few major questions: 
# 
# * Where kind of dataset does the list of known words come from?
# * Do they use all the words, or a selection of the words?
# * Where do the positive/negative scores come from?
# 
# Let's compare the tools we've used so far.
# 
# |technique|word source|word selection|scores|
# |---|---|---|---|
# |NLTK (VADER)|everywhere|hand-picked|internet people, word-by-word|
# |TextBlob|product reviews|hand-picked, mostly adjectives|internet people, word-by-word|
# |TextBlob + NaiveBayesAnalyzer|movie reviews|all words|automatic based on score|
# 
# A major thing that should jump out at you is **how different the sources are.**
# 
# While VADER focuses on content found everywhere, TextBlob's two options are specific to certain domains. The [original paper for VADER](http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf) passive-aggressively noted that VADER is effective at general use, but being trained on a specific domain can have benefits: 
# 
# > While some algorithms performed decently on test data from the specific domain for which it was expressly trained, they do not significantly outstrip the simple model we use.
# 
# They're basically saying, "if you train a model on words from a certain field, it will be good at sentiment in that certain field."

# ### Comparison chart
# 
# Because they're build differently, sentiment analysis tools don't always agree. Let's take a set of sentences and compare each analyzer's understanding of them.

# In[25]:


import pandas as pd
pd.set_option("display.max_colwidth", 200)

df = pd.DataFrame({'content': [
    "I love love love love this kitten",
    "I hate hate hate hate this keyboard",
    "I'm not sure how I feel about toast",
    "Did you see the baseball game yesterday?",
    "The package was delivered late and the contents were broken",
    "Trashy television shows are some of my favorites",
    "I'm seeing a Kubrick film tomorrow, I hear not so great things about it.",
    "I find chirping birds irritating, but I know I'm not the only one",
]})
df


# In[26]:


def get_scores(content):
    blob = TextBlob(content)
    nb_blob = blobber(content)
    sia_scores = sia.polarity_scores(content)
    
    return pd.Series({
        'content': content,
        'textblob': blob.sentiment.polarity,
        'textblob_bayes': nb_blob.sentiment.p_pos - nb_blob.sentiment.p_neg,
        'nltk': sia_scores['compound'],
    })

scores = df.content.apply(get_scores)
scores.style.background_gradient(cmap='RdYlGn', axis=None, low=0.4, high=0.4)


# Wow, those really don't agree with one another! Which one do you agree with the most? Did it get everything "right?"
# 
# While it seemed like magic to be able to plug a sentence into a sentiment analyzer and get a result back... maybe things aren't as magical as we thought.
# 
# #### Try ten sentences of your own
# 
# Just curious: can you make sentences that specifically "trick" one sentiment analysis tool or another?

# In[31]:


df = pd.DataFrame({'content': [
    "My cat is throwing up hairballs constantly and I'm worried",
    "It rained when I went to the botanical garden but it was still magical and delightful.",
    "I just got bit by multiple mosquitos. WTF.",
    "I don't really care for celeries, but I do like rhubarb",
    "Did you hear that the Bucks won the NBA Championship?",
    "They forgot my food and the delivery guy was late"
]})
df


# In[32]:


scores = df.content.apply(get_scores)
scores.style.background_gradient(cmap='RdYlGn', axis=None, low=-0.4, high=0.4)


# ## Review
# 
# **Sentiment analysis** is judging whether a piece of text has positive or negative emotion. We covered several tools for doing automatic sentiment analysis: **NLTK**, and two techniques inside of **TextBlob**.
# 
# Each tool uses a different data to determine what is positive and negative, and while some use **humans** to flag things as positive or negative, others use a automatic **machine learning**.
# 
# As a result of these differences, each tool can come up with very **different sentiment scores** for the same piece of text.

# ## Discussion topics
# 
# The first questions are about whether an analyzer can be applied in situations other than where it was trained. Among other things, you'll want to think about whether the language it was trained on is similar to the language you're using it on.
# 
# **Is it okay to use a sentiment analyzer built on product reviews to check the sentiment of tweets?** How about to check the sentiment of wine reviews?

# In[ ]:


# There are probably better sentiment analyzers for tweets. It's better if we can find analyzers built on social media posts.
# It's probably a good fit for wine review though, since it's built specifically for this kind of work.


# **Is it okay to use a sentiment analyzer trained on everything to check the sentiment of tweets?** How about to check the sentiment of wine reviews?

# In[ ]:


# Maybe? Tweets involve a lot of sarcasm and other linguistic nuances though, so it might not be a good fit.
# It might work better with wine reviews though? Since reviews usually use pretty straight forward language.


# **Let's say it's a night of political debates.** If I'm trying to report on whether people generally like or dislike what is happening throughout the debates, could I use these sorts of tools on tweets?
# 

# In[ ]:


# Maybe a sentiment analyzer built on tweets - and political tweets, if possible?
# Both political speech/opinions and tweets can be quite nuanced so I'd be careful about involving (unintentional?) bias.


# We're using the incredibly vague word "okay" on purpose, as there are varying levels of comfort depending on your sitaution. Are you doing this for preliminary research? Are you publishing the results in a journal, in a newspaper, in a report at work, in a public policy recommendation?
# 
# What if I tell you that the ideal of "I'd only use a sentiment analysis tool trained exactly for my specific domain" is both _rare and impractical?_ How comfortable do you feel with the output of sentiment analysis if that's the case?

# In[ ]:


# I feel like sentiment analysis can still be good for identifying trends and patterns. 
# It can probably still do a decent job helping us narrow our search and focus when we're doing preliminary research.
# But I would be more careful about publishing the results - maybe introduce some human review or vetting process if possible.
# I'd also think twice about the tool's weaknesses against the kind of content I want to analyze and weigh the pros and cons.
# Can we consider building our own tool? Do we have the resources?


# As we saw in the last section, **these tools don't always agree with one another, which might be problematic.**
# 
# * What might make them agree or disagree?
# * Do we think one is the "best?"
# * Can you think of any ways to test which one is the 'best' for our purposes?

# In[ ]:


# A few things that make them agree or disagree:
    # Whether they're trained on datasets with similar characteristics?
    # Whether there humans involved in the word polarity scorings?
    # Whether they are trained on a specific part of speech or on full sentences
    # Whether the machine learning is supervised or unsupervised
    
# Not sure if there's one that's universally "best." I feel like "best" is context dependent here.
# What's best is what most fits the content you're trying to analyze.

# Gather a sample of sentences that match our purposes and make a comparison table like we did above.
# Compare the scores against each other and against our own judgement or predictions.

