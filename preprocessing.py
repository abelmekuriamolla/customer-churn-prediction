import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(df, option):
    """
    This function is to cover all the preprocessing steps on the churn dataframe. It involves selecting important features, encoding categorical data, handling missing values,feature scaling and splitting the data
    """
    #Defining the map function
    def binary_map(feature):
        return feature.map({'Yes':1, 'No':0})

    # Encode binary categorical features
    binary_list = ['ATM','MobileBanking','InternetBanking']
    df[binary_list] = df[binary_list].apply(binary_map) 


    
    #Drop values based on operational options
    if (option == "Online"):
        columns = ['Age', 'Tenure', 'ATM', 'MobileBanking','InternetBanking', 'Sex_Female', 'Sex_Male', 'CivilStatus_Divorced', 'CivilStatus_Married', 
                   'CivilStatus_Single', 'Location_DAWRO', 'Location_GAMO', 'Location_GOFA', 'Location_HADIYA', 'Location_HALABA', 'Location_KEMBATA', 
                   'Location_KONSO', 'Location_OMO', 'Location_SILTE', 'Location_WOLAITA', 'ProductType_1320', 'ProductType_1328', 'ProductType_1335', 'ProductType_1336', 
                   'ProductType_1347', 'ProductType_1348', 'ProductType_1349', 'ProductType_1356', 'ProductType_1425', 'ProductType_1427']
        
        df = pd.get_dummies(df.reindex(columns=columns, fill_value=0))
    elif (option == "Batch"):
        pass
        df = df[['Sex','Age','CivilStatus','ProductType','Location','Tenure','ATM', 
                'MobileBanking','InternetBanking']]
        
        columns = ['Age', 'Tenure', 'ATM', 'MobileBanking','InternetBanking', 'Sex_Female', 'Sex_Male', 'CivilStatus_Divorced', 'CivilStatus_Married', 
                   'CivilStatus_Single', 'Location_DAWRO', 'Location_GAMO', 'Location_GOFA', 'Location_HADIYA', 'Location_HALABA', 'Location_KEMBATA', 
                   'Location_KONSO', 'Location_OMO', 'Location_SILTE', 'Location_WOLAITA', 'ProductType_1320', 'ProductType_1328', 'ProductType_1335', 'ProductType_1336', 
                   'ProductType_1347', 'ProductType_1348', 'ProductType_1349', 'ProductType_1356', 'ProductType_1425', 'ProductType_1427']
        
        df = pd.get_dummies(df.reindex(columns=columns, fill_value=0))

    else:
        print("Incorrect operational options")


    #feature scaling
    sc = StandardScaler()
    df['Tenure'] = sc.fit_transform(df[['Tenure']])
    df['Age'] = sc.fit_transform(df[['Age']])
    return df
        




