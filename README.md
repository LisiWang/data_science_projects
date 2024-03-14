Recently, I completed the intensive 8-week Data Science Fellowship at The Data Incubator (TDI), during which I worked on the following weekly projects:
## 1. Who are the most well-connected individuals and frequent pairs?
- **What I did:**
  - Parsed photo captions from New York Social Diary and constructed network of NYC's social elites
- **Data:** 1K+ captions from newyorksocialdiary.com
- **Main packages:** request, spacy, networkx
## 2. Can I predict star ratings of businesses?
- **What I did:**
  - Fine-tuned boosting and bagging ensemble models to predict star-ratings of businesses, utilizing KNeighborsRegressor, Ridge, RandomForestRegressor, and custom regressors
- **Data:** 37K+ businesses from Yelp open dataset
- **Main packages:** scikit-learn, pandas, numpy
## 3. 
- **What I did:** 
- **Data:** 
- **Main packages:** sql
## 4. Can I predict star ratings of businesses, with NLP?
- **What I did:**
 - Predicted star ratings of businesses from review texts by fine-tuning word vectorizers and SGDRegressor, increased coefficient of determination from 0.53 to 0.58
 - Analyzed word polarity (indicating 1- or 5-star reviews) using TfidfVectorizer and MultinomialNB
- **Data:** 253K+ reviews from Yelp open dataset
- **Main packages:** scikit-learn, pandas
## 5. How to model temperature over time?
- **What I did:**
  - Modelled time series of temperature grouped by 5 cities using custom Fourier transformer and LinearRegression
- **Data:** 392K+ observations
- **Main packages:** scikit-learn, pandas, numpy
## 6. Can I "profile" users?
- **What I did:**
  - Leveraged distributed computing to wrangle 10 gigabytes of Stack Exchange posts and users, in order to identify users' activity preference and active duration
- **Data:** 24M+ posts & 4M+ users from stackexchange.com
- **Main packages:** spark rdd, spark dataframe
## 7. Is your post going to be popular?
- **What I did:**
  - Fine-tuned HashingTF and LogisticRegression to predict if given post has most popular tags based on content
- **Data:** Same as 6
- **Main packages:** spark ml
## 8. Can deep learning help with labeling images?
- **What I did:**
  - built CNN and dense neural network on top of pre-trained deep neural network (inception model), improved image classification accuracy from 71% to 84%
- **Data:** 60K images from CIFAR-10 dataset
- **Main packages:** tensorflow, keras
