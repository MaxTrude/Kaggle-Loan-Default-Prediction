
# Loading libraries used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras import Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from keras.layers import Dropout
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


pd.set_option('display.max_columns', None)  # Show all data frame columns

n = 10000 # Limit rows read from CSV due to limitation in procession power


##
### DATA PREPROCESSING
##


data = pd.read_csv("Loan_Default.csv", nrows=10000) # Read data from CSV
data = data.drop("LoanID", axis=1) # Drop "LoanID" column because not used


# Generate lists for numerical and categorical columns in the dataset
numerical_columns = (data.drop('Default', axis=1)).select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

data_catSK_nFE = data.copy()    # Dataset with sklearn encoding and no artificial features
data_catMAN_nFE = data.copy()   # Dataset with manual encoding and no artificial features
data_catSK_FE = data.copy()     # Dataset with sklearn encoding and artificial features
data_catMAN_FE = data.copy()    # Dataset with manual encoding and artificial features

data_catSK_nFE.name = "data_catSK_nFE"
data_catMAN_nFE.name ="data_catMAN_nFE"
data_catSK_FE.name = "data_catSK_FE"
data_catMAN_FE.name = "data_catMAN_FE"

# Encode categorical values with sklearn built-in label encoder
le = LabelEncoder()
for col in categorical_columns:
    data_catSK_nFE[col+"_num"] = le.fit_transform(data[col])
    data_catSK_FE[col + "_num"] = le.fit_transform(data[col])

# Encoding not via sklearn, instead manually generating numerical, ORDINARY labels using assumptions form data analysis
Education_num = np.array([])
for i in range(n):
    if data['Education'][i] == "Bachelor's":
        Education_num = np.append(Education_num, 0)
    elif data['Education'][i] == "High School":
        Education_num = np.append(Education_num, 1)
    elif data['Education'][i] == "Master's":
        Education_num = np.append(Education_num, 3)
    elif data['Education'][i] == "PhD":
        Education_num = np.append(Education_num, 2)
data_catMAN_nFE["Education_num"] = Education_num
data_catMAN_FE["Education_num"] = Education_num

EmploymentType_num = np.array([])
for i in range(n):
    if data['EmploymentType'][i] == "Unemployed":
        EmploymentType_num = np.append(EmploymentType_num, 0)
    elif data['EmploymentType'][i] == "Part-time":
        EmploymentType_num = np.append(EmploymentType_num, 1)
    elif data['EmploymentType'][i] == "Self-employed":
        EmploymentType_num = np.append(EmploymentType_num, 2)
    elif data['EmploymentType'][i] == "Full-time":
        EmploymentType_num = np.append(EmploymentType_num, 3)
data_catMAN_nFE["EmploymentType_num"] = EmploymentType_num
data_catMAN_FE["EmploymentType_num"] = EmploymentType_num

MaritalStatus_num = np.array([])
for i in range(n):
    if data['MaritalStatus'][i] == "Divorced":
        MaritalStatus_num = np.append(MaritalStatus_num, 0)
    elif data['MaritalStatus'][i] == "Single":
        MaritalStatus_num = np.append(MaritalStatus_num, 1)
    elif data['MaritalStatus'][i] == "Married":
        MaritalStatus_num = np.append(MaritalStatus_num, 2)
data_catMAN_nFE["MaritalStatus_num"] = MaritalStatus_num
data_catMAN_FE["MaritalStatus_num"] = MaritalStatus_num

HasDependents_num = np.array([])
for i in range(n):
    if data['HasDependents'][i] == "Yes":
        HasDependents_num = np.append(HasDependents_num, 1)
    elif data['HasDependents'][i] == "No":
        HasDependents_num = np.append(HasDependents_num, 0)
data_catMAN_nFE["HasDependents_num"] = HasDependents_num
data_catMAN_FE["HasDependents_num"] = HasDependents_num

HasMortgage_num = np.array([])
for i in range(n):
    if data['HasMortgage'][i] == "Yes":
        HasMortgage_num = np.append(HasMortgage_num, 1)
    elif data['HasMortgage'][i] == "No":
        HasMortgage_num = np.append(HasMortgage_num, 0)
data_catMAN_nFE["HasMortgage_num"] = HasMortgage_num
data_catMAN_FE["HasMortgage_num"] = HasMortgage_num

LoanPurpose_num = np.array([])
for i in range(n):
    if data['LoanPurpose'][i] == "Other":
        LoanPurpose_num = np.append(LoanPurpose_num, 0)
    elif data['LoanPurpose'][i] == "Business":
        LoanPurpose_num = np.append(LoanPurpose_num, 1)
    elif data['LoanPurpose'][i] == "Education":
        LoanPurpose_num = np.append(LoanPurpose_num, 2)
    elif data['LoanPurpose'][i] == "Home":
        LoanPurpose_num = np.append(LoanPurpose_num, 3)
    elif data['LoanPurpose'][i] == "Auto":
        LoanPurpose_num = np.append(LoanPurpose_num, 4)
data_catMAN_nFE["LoanPurpose_num"] = LoanPurpose_num
data_catMAN_FE["LoanPurpose_num"] = LoanPurpose_num

HasCoSigner_num = np.array([])
for i in range(n):
    if data['HasCoSigner'][i] == "Yes":
        HasCoSigner_num = np.append(HasCoSigner_num, 1)
    elif data['HasCoSigner'][i] == "No":
        HasCoSigner_num = np.append(HasCoSigner_num, 0)
data_catMAN_nFE["HasCoSigner_num"] = HasCoSigner_num
data_catMAN_FE["HasCoSigner_num"] = HasCoSigner_num


#
# Feature Engineering
#
CreditScoreToInterestRate = np.array(data["CreditScore"])/np.array(data["InterestRate"])
#LoanToIncome= np.array(data["LoanAmount"])/np.array(data["Income"])
#IncomePerCreditLine = np.array(data["Income"])/np.array(data["NumCreditLines"])
#HighRiskLoan = ((data["InterestRate"] > 0.2) & (data["LoanAmount"] > data["LoanAmount"].median())).astype(int)
#Age_sq = np.array(data["Age"])**2
LongTermEmployment = np.array(data["MonthsEmployed"])**2
#LoanBurden = (np.array(data["LoanAmount"])*np.array(data["InterestRate"]))/np.array(data["Income"])

# Integrate synthetic features in dataset
data_catSK_FE["CreditScoreToInterestRate"] = CreditScoreToInterestRate
data_catMAN_FE["CreditScoreToInterestRate"] = CreditScoreToInterestRate
data_catSK_FE["LongTermEmployment"] = LongTermEmployment
data_catMAN_FE["LongTermEmployment"] = LongTermEmployment

# Generate datasets with smaller selection features
data_crop_catSK_nFE = (data_catSK_nFE.copy())[["Default", "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "InterestRate", "EmploymentType_num", "HasCoSigner_num"]]
data_crop_catSK_FE = (data_catSK_FE.copy())[["Default", "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "InterestRate", "EmploymentType_num", "HasCoSigner_num", "CreditScoreToInterestRate", "LongTermEmployment"]]
data_crop_catMAN_nFE = (data_catMAN_nFE.copy())[["Default", "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "InterestRate", "EmploymentType_num", "HasCoSigner_num"]]
data_crop_catMAN_FE = (data_catMAN_FE.copy())[["Default", "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "InterestRate", "EmploymentType_num", "HasCoSigner_num", "CreditScoreToInterestRate", "LongTermEmployment"]]

data_crop_catSK_nFE.name = "data_crop_catSK_nFE"
data_crop_catSK_FE.name = "data_crop_catSK_FE"
data_crop_catMAN_nFE.name = "data_crop_catMAN_nFE"
data_crop_catMAN_FE.name = "data_crop_catMAN_FE"





#
##
### DATA ANALYSIS
##
#

data_1 = data [data['Default']==1]
data_0 = data [data['Default']==0]
print(str(len(data_1)) + " default cases and " + str(len(data_0)) + " non-default cases\\")

numerical_columns = (data.drop("Default", axis=1)).select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

# Plotting distributions of categorical features: One for default, one for non-default
"""
for column in categorical_columns:
    plt.figure(figsize=(14, 6))
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    sns.countplot(data=data_1, x="EmploymentType", color='lightblue')
    plt.title(column + " for default cases", fontsize=20)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    sns.countplot(data=data_0, x="EmploymentType", color='lightblue')
    plt.title(column + " for non-default cases", fontsize=20)
    plt.tight_layout()
    plt.show()
"""

# Plotting histograms of numerical features: One for default, one for non-default
"""
for column in numerical_columns:
    plt.figure(figsize=(14, 6))
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.hist(data_1[column])
    plt.title(column + " for default cases", fontsize=20)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.hist(data_0[column])
    plt.title(column + " for non-default cases", fontsize=20)
    plt.tight_layout()
    plt.show()
"""


#
##
#### TEST OF DIFFERENT MODELS
##
#

data_frames = [data_catSK_nFE, data_catMAN_nFE, data_catSK_FE, data_catMAN_FE, data_crop_catSK_nFE, data_crop_catSK_FE,data_crop_catMAN_nFE, data_crop_catMAN_FE]
#data_frames = [data_catSK_nFE, data_catMAN_nFE]

result_df = pd.DataFrame({"Dataset":[], "Logistic Regression": [], "Random Forest": [], "Neural Network": []})

for data_f in data_frames:

    # Renew list of numerical columns after label encoding
    numerical_columns = data_f.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Scaling of numerical columns
    scaler = MinMaxScaler()
    data_f[numerical_columns] = scaler.fit_transform(data_f[numerical_columns])

    # Show correlation matrix und bar-plot of feature correlation with "Default"
    correlation_matrix = (data_f[numerical_columns]).corr()



    plt.figure(figsize=(14,14))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.1f', linewidths=0.5)
    plt.title("Heatmap for correleation between all features", fontsize=20)
    plt.tight_layout()
    plt.show()
    class_corr = (correlation_matrix.Default).sort_values(ascending=False)
    plt.barh(class_corr.index[1:], class_corr.values[1:], color="lightblue")
    plt.title("Correlation between features and default indicator")
    plt.tight_layout()
    plt.show()


    # Resampling (due to unbalanced data)
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_res, y_res = smote.fit_resample(data_f[numerical_columns].drop('Default', axis=1), data_f['Default'])

    # Train-Test Split of resampled data
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    """
    # Train-Test Split of NOT resampled data -> nearly all cases get classified as non-default resulting in ~88% accuracy
    X = data_f[numerical_columns].drop('Default', axis=1)
    y = data_f["Default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """


    #
    # LOGISTIC REGRESSION
    #

        # Model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    lr_acc = accuracy_score(y_test, y_pred)
        # Model Evaluation
    print("Logistic Regression on "  + data_f.name)
    print(f"Accuracy: {lr_acc}" + "\n")
    #print(classification_report(y_test, y_pred))
    coefficients = model.coef_[0]  # Array with coefficients
    features = X_train.columns

        # Feature-Coefficient data frame
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', key=abs, ascending=False)  # Nach Stärke sortieren

        # Bar-plot for coefficients of logistic regression
    plt.figure(figsize=(10, 8))
    sns.barplot(data=coef_df, x='Coefficient', y='Feature', color="lightblue")
    plt.title("Coefficients of Logistic Regression")
    plt.axvline(0, color='gray', linestyle='--')  # Null-Linie
    plt.tight_layout()
    plt.show()


    #
    # RANDOM FOREST
    #

        # Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Random Forest on " + data_f.name)
    rf_acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {rf_acc}" + "\n")
    #print(classification_report(y_test, y_pred))

        # Feature importance
    importances = model.feature_importances_
    features = X_train.columns
    forest_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    forest_df = forest_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=forest_df, x='Importance', y='Feature', color="lightblue")
    plt.title('Feature Importances in Random Forest Model')
    plt.tight_layout()
    plt.show()



    #
    # NEURAL NETWORK
    #

        # Model
    model = Sequential(
        [
            Dense(units=128, activation="relu"),
            Dense(units=64, activation="relu"),
            Dropout(0.3),
            Dense(units=8, activation="relu"),
            Dropout(0.3),
            Dense(units=1, activation="sigmoid") #"linear" statt "softmax" für mehr numerische Stabilität
        ]
    )
    model.compile(loss = BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])

        # Model fitting with option of class-dependent weights
    class_weights = {0: 1, 1: 1}
    model.fit(X_train, y_train, epochs=10, class_weight=class_weights)

        # Prediction on test data
    y_pred = model.predict(X_test)

        # Analysis of model's prediciton
    y_pred = (y_pred > 0.5).astype(int)
    print("Neural Network on " + data_f.name)
    nn_acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {nn_acc}")
    print(classification_report(y_test, y_pred))

        # Feature importance
    from sklearn.metrics import accuracy_score
    baseline_acc = accuracy_score(y_test, model.predict(X_test) > 0.5)
    importances = []
    for i, col in enumerate(X_test.columns):
        X_permuted = X_test.copy()
        X_permuted[col] = np.random.permutation(X_permuted[col])
        permuted_acc = accuracy_score(y_test, model.predict(X_permuted) > 0.5)
        drop_in_acc = baseline_acc - permuted_acc
        importances.append(drop_in_acc)

    importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Bar-plot for importance
    plt.figure(figsize=(10, 8))
    sns.barplot(data=coef_df, x='Coefficient', y='Feature', color="lightblue")
    plt.title("Feature importance in Neural Network")
    plt.axvline(0, color='gray', linestyle='--')  # Null-Linie
    plt.tight_layout()
    plt.show()


    result_df.loc[len(result_df)] = [data_f.name, round(lr_acc, 2), round(rf_acc,2), round(nn_acc,2)]



# Table with resulting accuracy
print(result_df)
fig, ax = plt.subplots(figsize=(10, len(result_df)*0.5))  # dynamische Höhe
ax.axis('off')  # keine Achsen

table = ax.table(cellText=result_df.values,
                 colLabels=result_df.columns,
                 loc='center',
                 cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)  # Breite, Höhe der Zellen
plt.show()