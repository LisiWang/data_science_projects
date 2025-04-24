The following projects are organized as **what I did, data, main packages, *either* visualization *or* code snippet**: \

## SQL: Which violations are more common for which business types?
- **What I did:**
  - Analyzed compliance audits using SQL and CTEs to normalize conditional probabilities of violations by business type and location, enabling meaningful comparisons.
```math
Normalized\ conditional\ probability = {P(Specific\ violation\ |\ Specific\ buisiness\ type) \over P(Specific\ violation\ |\ All\ business\ type)}
```
- **Data:** 530K+ compliance audits from [NYC open data](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/about_data)
- **Main packages:** sql
- **[Code snippet](https://github.com/LisiWang/tdi_weekly_projects/blob/114b06a0068ce0a84d3275d9c6f41798e95e7f0b/code_snippets/sql.ipynb)**

## PySpark: "Profiling" users
- **What I did:**
  - Used Spark to analyze 10GB of online forum log data, uncovering patterns in user engagement metrics such as preference and retention rate.
- **Data:** 24M+ posts & 4M+ users from [stackexchange.com](https://archive.org/details/stackexchange)
- **Main packages:** pyspark rdd, pyspark dataframe
- **[Code snippet](https://github.com/LisiWang/tdi_weekly_projects/blob/114b06a0068ce0a84d3275d9c6f41798e95e7f0b/code_snippets/spark_df.ipynb)**

## ML: Predicting star ratings of businesses
- **What I did:**
  - Fine-tuned boosting and bagging ensemble models to predict star-ratings of businesses, utilizing KNN, ridge, random forest, and custom regressors
- **Data:** 37K+ businesses from [Yelp open dataset](https://www.yelp.com/dataset)
- **Main packages:** scikit-learn, pandas, numpy
- **Visualization:**
<p align="center">
<img src="https://github.com/LisiWang/tdi_weekly_projects/blob/114b06a0068ce0a84d3275d9c6f41798e95e7f0b/visualizations/ml.png" height="240">
</p>

## NLP: Predicting star ratings of businesses
- **What I did:**
  - Predicted star ratings of businesses from review texts by fine-tuning word vectorizers and stochastic gradient descent regressor, increased R<sup>2</sup> (coefficient of determination) from 0.53 to 0.58
  - Analyzed word polarity (indicating 1- or 5-star reviews) using TF-IDF vectorizer and multinomial Naive Bayes classifier
- **Data:** 253K+ reviews from [Yelp open dataset](https://www.yelp.com/dataset)
- **Main packages:** scikit-learn, pandas, numpy
- **Visualization:**
<p align="center">
<img src="https://github.com/LisiWang/tdi_weekly_projects/blob/114b06a0068ce0a84d3275d9c6f41798e95e7f0b/visualizations/nlp.png" height="300">
</p>

## Deep Learning: Improving neural networks for image classification?
- **What I did:**
  - Built CNN and transfer learning model ([inception](https://github.com/tensorflow/tpu/tree/906be5267106a72d51d682d6fda15210118840cf/models/experimental/inception) deep learning + dense neural network), improved image classification accuracy from 71% to 84%
- **Data:** 60K images from [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Main packages:** tensorflow, keras, numpy, scikit-learn
<p align="center">
<img src="https://github.com/LisiWang/tdi_weekly_projects/blob/114b06a0068ce0a84d3275d9c6f41798e95e7f0b/visualizations/tf.png" height="300">
</p>

- **[Code snippet](https://github.com/LisiWang/tdi_weekly_projects/blob/114b06a0068ce0a84d3275d9c6f41798e95e7f0b/code_snippets/tf.ipynb)**
## Graph: Who are the most well-connected individuals?
- **What I did:**
  - Parsed photo captions from New York Social Diary and constructed network of NYC's social elites, to identify most well-connected individuals and frequent pairs
- **Data:** 1K+ captions from [newyorksocialdiary.com](https://www.newyorksocialdiary.com/)
- **Main packages:** request, spacy, networkx
- **Visualization:**
<p align="center">
<img src="https://github.com/LisiWang/tdi_weekly_projects/blob/114b06a0068ce0a84d3275d9c6f41798e95e7f0b/visualizations/graph.png" height="400">
</p>

## Time series: Modeling temperature over time
- **What I did:**
  - Modeled time series of temperature grouped by 5 cities using custom Fourier transformer and linear regression
- **Data:** 392K+ observations
- **Main packages:** scikit-learn, pandas, numpy
- **Visualization:**
<p align="center">
<img src="https://github.com/LisiWang/tdi_weekly_projects/blob/114b06a0068ce0a84d3275d9c6f41798e95e7f0b/visualizations/ts.png" height="300">
</p>

In case you're curious about my **capstone project**, please go to [this repo](https://github.com/LisiWang/tdi_capstone_project.git).
