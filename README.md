# Payment-Gateway_Fraud-Detection_Ind-Avenue
Note : This Repository consists files of the Project -Detecting fraud for transactions in a payment gateway which was held as a Mid Term Hackathon Competetion as a part of my PGP Data Science Course @ InsofeMid Term 


(I) PROBLEM STATEMENT :
Detecting fraud for transactions in a payment gateway -
A new disruptive payment gateway start-up, ‘IndAvenue’, has started gaining traction due to its extremely low processing fees for handling online vendors’ digital payments.
This strategy has led to very low costs of acquiring new vendors. Unfortunately, due to the cheap processing fees, the company was not able to build and deploy a robust
and fast fraud detection system. Consequently, a lot of the vendors have accumulated significant economic burden due to handling fraudulent transactions on their platforms. This has resulted in a significant number of current clients leaving IndAvenue’s payment gateway platform for more expensive yet reliable payment gateway companies.
The company’s data engineers curated a dataset that they believe follows the real world distribution of transactions on their payment gateway. The company hired
Insofe and provided it with the dataset, to create a fast and robust AI based model that can detect and prevent fraudulent transactions on its payment gateway.
They have provided you with the dataset that has the `is_fraud` column, which encodes the information whether a transaction was fraudulent or not. In this hackathon, you will now have to use this curated data to create a machine learning model that will be able to predict the `is_fraud` column.


(II) STRATEGY/APPROACH : I have preprocessed the data, performed EDA,Data Prep, visualized the data and validated the insights ,Applied SMOTE for oversampling as the data is unbalanced and applied Logistic regression, Decision Tree Classifier, Random forest, GBM algorithms to predict the fraudulent transactions.

Evaluated the Model using F1 score as a metric and curated a submission file using the transaction no and the target attribute

(III) TOOLS AND TECHNOLOGY STACK : Jupyter Notebooks, Google Colab , MS Excel, Python Scripting.

