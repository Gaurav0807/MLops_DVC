import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def createDataFrame():
    data = {
        "id": [1,2,3,4,5,6,7,8,9,10],
        "review": [
            "Great Food and ambiance",
            "Teriable Service",
            "Amazing Experience",
            "Food was cold",
            "Loved the desserts",
            "Not worth the money",
            "Excellent customer service",
            "The place was too crowded",
            "Best restaurant in town",
            "Average Experience"
        ]
    }
    
    df = pd.DataFrame(data)
    return df


def save_dataframe(df):
    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_csv("data/data.csv", index=False)
    print("File saved")

def process_data(k):

    df = pd.read_csv("data/data.csv")

    vectorizer = CountVectorizer(max_features=k)
    vectorized_data = vectorizer.fit_transform(df["review"])
    feature_name = vectorizer.get_feature_names_out()

    vectorizer_df = pd.DataFrame(vectorized_data.toarray(), columns=feature_name)
    processed_data = pd.concat([df,vectorizer_df],axis=1)

    processed_data.to_csv("data/processed_data.csv", index=False)

    print("Data Saved")
    return processed_data

if __name__ == "__main__":

    df = createDataFrame()

    save_dataframe(df)

    k = 5
    processed_df = process_data(k)
    print(f"Data Shape :- {df.shape}")
    print(f"Processed Data Shape :- {processed_df.shape}")


