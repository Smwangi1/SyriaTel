# *KEEPING CUSTOMERS CONNECTED - AND NOT DISCONNECTED!* 
 ## THE SYRIATEL ANALYSIS

# 1.BUSINESS UNDERSTANDING

## **1.1 BUSINESS OVERVIEW**

According to this [article](https://www.sciencedirect.com/topics/social-sciences/telecommunications-industry) published in 2011, Telecommunications company is an organization that provides services for long distance communication. They do this by building and mainatining  the physical networks, like cell towers, that transmit signals to individuals and businesses.These companies facilitate essential services like accessing the internet, making phone calls and sending messages. They make money through customer subscriptions and usage fees for these services.SyriaTel is a telecom company that provides call,text and data services to customers. 
One advantage of working with in the telecommunication sector is that it is a high-performing sector that contributes to economic growth, potentially increasing returns for investors. Telecommunication is also an essential service with steady demand, making it stable and a valuable industry to be part of.
However, the telecom industry is highly competitive and customers can easily switch to other providers if they're dissatisfied. This creates a high risk of customer churn, which can reduce revenue and can discourage investor confidence if not properly managed.

<img src="telecomm.webp" alt="Telecomm " width="600">





## **1.2 PROBLEM STATEMENT**
SyriaTel is losing customers to competitors, by analysing customer data, we can predict churn and uncover the reasons why customers leave, so SyriaTel can take action to reduce churn and improve customer retention.

This is costly because:

*Revenue loss:* Each customer lost means recurring revenue lost.

*High acquisition cost:* It is more expensive to acquire a new customer than to retain an exsisting one.

*Competitive pressure:* In a competetive market, reducing churn is critical for survival and growth.

If we can predict which customers are likely to leave, SyriaTel can take action early eg. giving offers, improving services,or solving problems to make those customers stay.

So the goal is to reduce churn and keep loyal customers.


## **1.3 BUSINESS OBJECTIVES**

 ## 1.3.1 *Main objective:*
To create a succesful binary classification model that predicts customers likelihood to churn.


 ## 1.3.2 *Specific objectives:*

1. To develop a model that predicts whether a customer will churn or stay.
2. To predict customer churn and provide insights that help SyriaTel keep its customers and reduce revenue loss.
3. To identify the key factors eg. call charges, service quality or customer complaints that influence the probability of a customer to churn or not to churn.
4. To provide insights that SyriaTel can use to design strategies for reducing churn and improving customer satisfaction.
5. To determine the state with the highest churning rate.



 ## 1.3.3*Research questions*
1. What is the best performing binary classification to use for prediction?
2. Can we accurately predict which syriaTel customers are likely to churn using their demographic and usage data?
3. What are the main factors that influence customer churn?
4. How can SyriaTel use the model's prediction and insights to design strategies that reduce churn and retain more customers?
5. What is the state with the highest churning rate?

## **1.4 SUCCESS CRITERIA**

 ***Model performance***
 
The churn prediction model achieves a good level of balance between recall and ROC-AUC score inorder to correctly identifying customers who churn.
 
 ***Insights gained***
 
The analysis clearly identifies the key factors that contribute to churn eg. high call charges and frequent complains.
 
 ***Business value***
 
SyriaTel can use the model's results to take practical actions, such as designing loyalty offers or improving customer service which can help improve customer churn.


# 2. DATA UNDERSTANDING
The Syria Tel customer churn dataset we are working with is from [Kaggle](https://www.kaggle.com/becksddf/churn-in-telecoms-dataset).Our data is on Syria Tel which is a telecommunication conmpany, it had a total of 21 columns and 3333 rows after data cleaning we decided to work with the coloumns below: where `churn` is our dependent varaible.
    
`state` – U.S. state where the customer lives.
    
`account length `– Number of days the customer has had the account.
    
`area code` – Telephone area code.
    
`phone number` – Customer’s phone number (serves as an identifier, not useful for prediction).
    
`international plan`– Whether the customer has an international calling plan (yes/no).
    
`log_vmail_messages` – Number of voicemail messages the customer has.
    
`customer service calls` – Number of calls made to customer service.
    
**`churn`** – Whether the customer left the company (True = churned, False = stayed). which is our dependent variable**
    
`total_calls` - The total number of calls.
    
`total_minutes` - The total number number of minutes for all calls.
    
`total_charge` - The total charges for all calls.

     
We merged the columns `total day minutes` ,`total eve minutes` and `total night minutes` into one column named `total_minutes`. We are also merged `total day calls` , `total eve calls` and `total night calls` into one column named `total_calls`. The columns `total day charge`, `total eve charge`,  and`total night charge` are were also being merged to become one column called `total_charge`.

Since our dataset didn't have any missing values we didn't have to drop any null or fill for missing values.

We dropped the `phone number` column since it was not useful in our prediction.

We also needed to check for categorical data in our dataset so that we can perform **one hot encoding** which is an important step for us to make predictions and create Machine learning models.

 # 3 Feature engineering

Under this we checked how different columns correlated with each other before we decided on what features to use so we needed to conduct feature engineering on the columns.

# 4. EXPLANATORY DATA ANALYSIS

We did a bit of explanatory data analysis before we moved to building our models. e.g.

## 4.1 Top 5 and Bottom 5 states with highest churn rate.


```python
# Group by state and churn counts
state_churn = df_encoded.groupby(["state", "churn"]).size().unstack(fill_value=0)

# Add churn rate per state
state_churn["churn_rate"] = state_churn[1] / (state_churn[0] + state_churn[1])

# Sort by churn rate (descending)
highest_churn = state_churn.sort_values(by="churn_rate", ascending=False).head(5)

least_churn = state_churn.sort_values(by ="churn_rate", ascending=False).tail(5)

print("Top 5 states with the highest churn rate:")
print(highest_churn)# show top 10 states with highest churn rate
print("Top 5 states with the least churn rate:")
print(least_churn)

```

    Top 5 states with the highest churn rate:
    churn   0   1  churn_rate
    state                    
    NJ     50  18    0.264706
    CA     25   9    0.264706
    TX     54  18    0.250000
    MD     53  17    0.242857
    SC     46  14    0.233333
    Top 5 states with the least churn rate:
    churn   0  1  churn_rate
    state                   
    IA     41  3    0.068182
    VA     72  5    0.064935
    AZ     60  4    0.062500
    AK     49  3    0.057692
    HI     50  3    0.056604
    
This would guide the company to know where it's loyal customers are and where they have a stronger market.


# 5. MODELLING

## 5.1 BASELINE MODEL

### 5.1.1 LOGISTIC REGRESSION

We moved to creating models,specifically a logistic regression model because the problem we were trying to answer is a binary classification and we were also trying to answer the question **what is the probability of a customer to churn or not to churn?** depending on various features.

Before moving to the modelling bit of things we first had to do **feature selection** and also look at our predictor variable `churn`, So as to create our baseline model.


## 5.2 LOGISTIC REGRESSION WITH ALL FEATURES.

We wanted to check whether adding more information improves the model compared to the baseline.For features we are using all the predictores available in the dataset excluding the target variable. We first onehotencode the multi-categorical variable state to have a smooth flow.


## 5.3 DECISIONTREE CLASSIFIER

We built another model to perform classification ,in this case a DecisionTree Classifier .

# 6.EVALUATION

In this section we will be evaluating our models to determine which performs better at predicting churning customers. We will compare the models and ultimately choose the one that performs better as our baseline model of recommendation.Let's begin.

We will be using Recall and ROC-AUC as the metric of success of our model.we will be using:

**Recall** 
*  Recall measures how many actual churners the model correctly identifies.
*  In churn prediction, missing a churner is costly, because it means losing a customer and revenue.
*  By optimizing high recall, we ensure the model captures most at-risk customers, even if it occasionally flags a few non-churners.

**ROC-AUC**
* measures the model’s ability to discriminate between churners and non-churners across all thresholds.
* ROC-AUC is threshold-independent, so it evaluates the model’s overall ranking ability.
* A high ROC-AUC means the model is reliable in assigning higher churn probabilities to churners than to non-churners, which is critical for making informed business decisions.

Togther they align with our business objectives and the problem we are tyrying to solve.

The Decision Tree is performing much better across all metrics than the baseline logistic regression.

**Recall**: Slightly improved from 0.753 to 0.808 meaning the tree catches more actual churners.

**ROC-AUC**: Higher AUC of 0.90 means the tree has a much better ability to discriminate churners from non-churners overall

The Decision Tree clearly outperforms the baseline logistic regression on this dataset.

We compared the models ROC-AUC and Recall across the models.

Our model correctly identifies 84.8% of the actual churners.A high recall will ensure we catch more churners and this important in churn prediction because missing churners (false negatives) can lead to lost revenue.ROC-AUC measures the model’s ability to discriminate between churners and non-churners across all possible thresholds. Our score of 0.917 indicates that if you randomly pick a churner and a non-churner, the model assigns a higher probability of churn to the churner 91.7% of the time.
We can conclude that This model is very good at separating churners from non-churners compared to our baseline model and other models too.


  ## HOW FEATURES INFLUENCING CHURN
This are the features that increase rate of churning in the model we seek to deploy:

**total charge** Customers with higher total charges are more likely to churn.High spending may indicate dissatisfaction with value or plan costs.

**customer service calls** Frequent calls to customer service strongly predict churn. Likely reflects unresolved issues or poor service experience.

**total_minute** Customers with higher total minutes usage may be at risk; possibly they are testing services or comparing alternatives.

**international plan** Having an international plan increases churn risk. Possibly due to cost or underuse of the plan.

**area code** Area code does not influence churn prediction.

**state** Geographic location contributes very little.

## **NOTE :**

Based on our evaluation, the Random Forest model achieves the best balance of high recall (0.85) and ROC-AUC (0.92) with minimal overfitting, making it the most suitable model for deployment in predicting customer churn.

# 7.CONCLUSION
1. New Jersey (NJ) and California (CA) are the states with the highest churning rate at 26.5% churn rate.
2. Hawaii (HI) and Alaska (AK) have the most loyal customers with a low rate of 5.7% and 5.8%  churn rate respectively.
3. The best performing model has a recall of 84.5% and an ROC-AUC score of 91.7% .
4. High charges is the highest feature influencing churn at 0.305
5. Random Forest model is because it achieves the best balance of high recall (0.85) and ROC-AUC (0.92) with minimal overfitting.
6. Random forest model predicted a total of 963 out 1000 correctly .

# 8. RECCOMENDATIONS
1. Stakeholders should focus on states with highest churn NJ(New jersey) and CA(Carlifonia) with special offers, loyalty programs, or improved customer service inorder to retain the customers.
2. High charges, frequent customer service calls and international plans are the key factors driving churn. By targeting retention efforts to these customers, the company can maximize revenue retention.
3. Stakeholders should de-priotize  Voicemail usage, account length, state, and area code because they have minimal predictive value when working on interventions.
4. Stakeholders should investigate complaints or service issues in high-churn states to reduce dissatisfaction and also pay attention to areas that are reported by customer complaints to increase customer satisfaction .
5. Stakeholders should relocate resources to higher-risk states such as New Jersey, California ,Texas among others since churn is low in those states and replicate strategies used in those states to improve others.
