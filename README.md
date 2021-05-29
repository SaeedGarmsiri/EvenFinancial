# EvenFinancial

This Code has been developed as a home assignment for the Data Scientist position at the Even Financial

## In order to run the code please go through the following steps:

1. Install the required libraries using the command 
   * `pip install -r requirements.txt`
3. Creat an empty schema on MySql and name it **even_financial**
4. First of all, run the **SchemaManager.py** python file. This file will create MySql tables and insert data in to them
5. Run the **DataProcessor.py** python file. This file handles pre-processing part of the project including cleaning data and feature engineering
6. Run the **LogisticModel.py** to fit the Logistic Regression on the data and dump the trained model

For the next step, we are going to test our model by sending two post requests to get the prediction result for the Jsonified features. there are two post methods in the **ModelAPI.py** file. One of them called _predict_single_ which receives a Json file of a single Lead and a single offer. another one is _predict_batch_ that receives a request that contains Jsonified features of a lead and its offers.

### To send Json type features to the model please go through the following steps:
1. Run the **ModelAPI.py** python file.
2. For testing single offer request, In the **Post Man** application:
    * Creat a new Post request like this: `http://127.0.0.1:5000/predict_single`
    * Add the bellow Json mock object in the body section of the request:
        ```json
        [
            {
                "loan_purpose_auto": 0,
                "loan_purpose_auto_purchase": 0,
                "loan_purpose_auto_refinance": 0,
                "loan_purpose_baby": 0,
                "loan_purpose_boat": 0,
                "loan_purpose_business": 0,
                "loan_purpose_credit_card_refi": 0,
                "loan_purpose_debt_consolidation": 1,
                "loan_purpose_home_improvement": 0,
                "loan_purpose_household_expenses": 0,
                "loan_purpose_large_purchases": 0,
                "loan_purpose_medical_dental": 0,
                "loan_purpose_moving_relocation": 0,
                "loan_purpose_other": 0,
                "loan_purpose_special_occasion": 0,
                "loan_purpose_student_loan": 0,
                "loan_purpose_taxes": 0,
                "loan_purpose_vacation": 0,
                "loan_purpose_wedding": 0,
                "apr": 199.0,
                "requested": 500.0,
                "credit": 3,
                "annual_income": 72000.0
            }
        ]
        ```
1. For testing multiple offers request, In the Post Man application:
    * Creat a new Post request like this: `http://127.0.0.1:5000/predict_batch`
    * Add the bellow Json mock object in the body section of the request:
       ```json
        [
          {
              "loan_purpose_auto": 0,
              "loan_purpose_auto_purchase": 0,
              "loan_purpose_auto_refinance": 0,
              "loan_purpose_baby": 0,
              "loan_purpose_boat": 0,
              "loan_purpose_business": 0,
              "loan_purpose_credit_card_refi": 0,
              "loan_purpose_debt_consolidation": 1,
              "loan_purpose_home_improvement": 0,
              "loan_purpose_household_expenses": 0,
              "loan_purpose_large_purchases": 0,
              "loan_purpose_medical_dental": 0,
              "loan_purpose_moving_relocation": 0,
              "loan_purpose_other": 0,
              "loan_purpose_special_occasion": 0,
              "loan_purpose_student_loan": 0,
              "loan_purpose_taxes": 0,
              "loan_purpose_vacation": 0,
              "loan_purpose_wedding": 0,
              "apr": 125.0,
              "requested": 2000.0,
              "credit": 3,
              "annual_income": 30000.0
          },
          {
              "loan_purpose_auto": 0,
              "loan_purpose_auto_purchase": 0,
              "loan_purpose_auto_refinance": 0,
              "loan_purpose_baby": 0,
              "loan_purpose_boat": 0,
              "loan_purpose_business": 0,
              "loan_purpose_credit_card_refi": 0,
              "loan_purpose_debt_consolidation": 1,
              "loan_purpose_home_improvement": 0,
              "loan_purpose_household_expenses": 0,
              "loan_purpose_large_purchases": 0,
              "loan_purpose_medical_dental": 0,
              "loan_purpose_moving_relocation": 0,
              "loan_purpose_other": 0,
              "loan_purpose_special_occasion": 0,
              "loan_purpose_student_loan": 0,
              "loan_purpose_taxes": 0,
              "loan_purpose_vacation": 0,
              "loan_purpose_wedding": 0,
              "apr": 125.0,
              "requested": 2000.0,
              "credit": 3,
              "annual_income": 30000.0
          },
          {
              "loan_purpose_auto": 0,
              "loan_purpose_auto_purchase": 0,
              "loan_purpose_auto_refinance": 0,
              "loan_purpose_baby": 0,
              "loan_purpose_boat": 0,
              "loan_purpose_business": 0,
              "loan_purpose_credit_card_refi": 0,
              "loan_purpose_debt_consolidation": 1,
              "loan_purpose_home_improvement": 0,
              "loan_purpose_household_expenses": 0,
              "loan_purpose_large_purchases": 0,
              "loan_purpose_medical_dental": 0,
              "loan_purpose_moving_relocation": 0,
              "loan_purpose_other": 0,
              "loan_purpose_special_occasion": 0,
              "loan_purpose_student_loan": 0,
              "loan_purpose_taxes": 0,
              "loan_purpose_vacation": 0,
              "loan_purpose_wedding": 0,
              "apr": 125.0,
              "requested": 2000.0,
              "credit": 3,
              "annual_income": 30000.0
          },
          {
              "loan_purpose_auto": 0,
              "loan_purpose_auto_purchase": 0,
              "loan_purpose_auto_refinance": 0,
              "loan_purpose_baby": 0,
              "loan_purpose_boat": 0,
              "loan_purpose_business": 0,
              "loan_purpose_credit_card_refi": 0,
              "loan_purpose_debt_consolidation": 1,
              "loan_purpose_home_improvement": 0,
              "loan_purpose_household_expenses": 0,
              "loan_purpose_large_purchases": 0,
              "loan_purpose_medical_dental": 0,
              "loan_purpose_moving_relocation": 0,
              "loan_purpose_other": 0,
              "loan_purpose_special_occasion": 0,
              "loan_purpose_student_loan": 0,
              "loan_purpose_taxes": 0,
              "loan_purpose_vacation": 0,
              "loan_purpose_wedding": 0,
              "apr": 125.0,
              "requested": 2000.0,
              "credit": 3,
              "annual_income": 30000.0
          },
          {
              "loan_purpose_auto": 0,
              "loan_purpose_auto_purchase": 0,
              "loan_purpose_auto_refinance": 0,
              "loan_purpose_baby": 0,
              "loan_purpose_boat": 0,
              "loan_purpose_business": 0,
              "loan_purpose_credit_card_refi": 0,
              "loan_purpose_debt_consolidation": 1,
              "loan_purpose_home_improvement": 0,
              "loan_purpose_household_expenses": 0,
              "loan_purpose_large_purchases": 0,
              "loan_purpose_medical_dental": 0,
              "loan_purpose_moving_relocation": 0,
              "loan_purpose_other": 0,
              "loan_purpose_special_occasion": 0,
              "loan_purpose_student_loan": 0,
              "loan_purpose_taxes": 0,
              "loan_purpose_vacation": 0,
              "loan_purpose_wedding": 0,
              "apr": 125.0,
              "requested": 2000.0,
              "credit": 3,
              "annual_income": 30000.0
          },
          {
              "loan_purpose_auto": 0,
              "loan_purpose_auto_purchase": 0,
              "loan_purpose_auto_refinance": 0,
              "loan_purpose_baby": 0,
              "loan_purpose_boat": 0,
              "loan_purpose_business": 0,
              "loan_purpose_credit_card_refi": 0,
              "loan_purpose_debt_consolidation": 1,
              "loan_purpose_home_improvement": 0,
              "loan_purpose_household_expenses": 0,
              "loan_purpose_large_purchases": 0,
              "loan_purpose_medical_dental": 0,
              "loan_purpose_moving_relocation": 0,
              "loan_purpose_other": 0,
              "loan_purpose_special_occasion": 0,
              "loan_purpose_student_loan": 0,
              "loan_purpose_taxes": 0,
              "loan_purpose_vacation": 0,
              "loan_purpose_wedding": 0,
              "apr": 249.0,
              "requested": 2000.0,
              "credit": 3,
              "annual_income": 30000.0
          }
      ]
      ```
      
 The result of the model is a list the 0s and 1s. 0 for the offer which is not predicted to be clicked and 1 in opposite. 
      
