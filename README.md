In 2022, the Supreme Court's decision in Dobbs v. Jackson overturned its landmark Roe v. Wade decision, not only marking a swift stride backward in the progression of women's rights but also highlighting a striking volatility in opinions surrounding abortion. In the time since, nearly half of U.S. states have imposed [severe restrictions or even outright bans on abortion](https://www.nytimes.com/interactive/2024/us/abortion-laws-roe-v-wade.html). Analyzing the evolving sentiment towards abortion offers a valuable opportunity to understand the ideological shifts that have taken place around these judicial and legislative changes.

To do so, we identified 14,019 abortion-related speeches from the [Congressional Record's corpus](https://www.congress.gov/congressional-record) of the over 18 million congressional speeches given since 1873 \cite{uscr}. We selected a random subset ($N = 103$) of the abortion-related speeches and labeled them as pro-life (0) or pro-choice (1) to create a labeled dataset. We apply a number of text processing techniques (Bag-of-Words (BoW), TF-IDF, and Word2Vec representations) to the labeled dataset and use it to train models (Logistic Regression, SVM, Naive Bayes, Random Forest, AdaBoost, and shallow Neural Networks). 

A Neural Network with 2 Dense ReLu layers on 1-gram TF-IDF features achieved a 100% accuracy rate on the labeled dataset, and we used it to automate the annotation of the rest of the speeches, enabling us to conduct analysis on trends in sentiment toward abortion over time. We found that for most of history, Congress has been pro-life and polarized. However, over time, it is becoming more pro-choice and less polarized, but sentiment is constantly, rapidly fluctuating. Additionally, congressional and public opinion generally don’t align (the public is more pro-choice), but they are trending in the same direction (more pro-choice).

In the future, we would want to expand the labeled dataset to see if the model still generalizes well, test out alternative methods for filtering speeches (e.g. topic modeling), and incorporate other data sources (e.g. campaign speeches, legislative records).

