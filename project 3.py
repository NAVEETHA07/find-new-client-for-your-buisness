import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import time

# Web Scraping
def scrape_crunchbase_data():
    url = "https://www.crunchbase.com/search/organization.companies"
    params = {
        "query": "recently funded companies",
        "pageSize": 100,
        "page": 1
    }

    response = requests.get(url, params=params)
    soup = BeautifulSoup(response.content, "html.parser")

    data = []
    for company in soup.find_all("div", class_="cb-row cb-company-row"):
        name = company.find("a", class_="name-link").text.strip()
        industry = company.find("div", class_="category-list").text.strip()
        location = company.find("div", class_="location-list").text.strip()
        funding = company.find("div", class_="funding-total-indicator").text.strip()
        data.append({"Name": name, "Industry": industry, "Location": location, "Funding": funding})

    return pd.DataFrame(data)

# Data Cleaning and Analysis
def clean_and_analyze_data(df):
    df["Funding"] = df["Funding"].str.replace("$", "").str.replace(",", "").astype(float)
    df["Location"] = df["Location"].str.split(",").str[0]
    df["Industry"] = df["Industry"].str.split(",").str[0]

    print(df.describe())
    print(df.groupby("Industry")["Funding"].mean())
    print(df.groupby("Location")["Funding"].mean())

    return df

# Feature Engineering
def engineer_features(df):
    df["Company_Age"] = 2023 - df["Founded_Year"]
    df["Funding_per_Employee"] = df["Funding"] / df["Number_of_Employees"]
    df["Industry_Growth_Rate"] = df.groupby("Industry")["Funding"].pct_change().mean()
    return df

# Model Development
def train_outsourcing_prediction_model(df):
    X = df[["Company_Age", "Funding_per_Employee", "Industry_Growth_Rate"]]
    y = df["Outsourcing_Likelihood"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Accuracy:", model.score(X_test, y_test))
    return model

# Deployment and Ongoing Monitoring
def predict_outsourcing_likelihood(model, company_age, funding_per_employee, industry_growth_rate):
    data = pd.DataFrame({
        "Company_Age": [company_age],
        "Funding_per_Employee": [funding_per_employee],
        "Industry_Growth_Rate": [industry_growth_rate]
    })
    return model.predict_proba(data)[0][1]

def monitor_model_performance(model):
    while True:
        new_data = pd.read_csv("new_company_data.csv")
        new_predictions = new_data.apply(
            lambda row: predict_outsourcing_likelihood(
                model, row["Company_Age"], row["Funding_per_Employee"], row["Industry_Growth_Rate"]
            ),
            axis=1
        )
        new_data["Predicted_Outsourcing_Likelihood"] = new_predictions
        new_data.to_csv("updated_company_data.csv", index=False)
        # Add any additional monitoring or retraining logic here
        time.sleep(86400)  # Wait for 24 hours before checking for new data

# Main function
def main():
    # Web Scraping
    df = scrape_crunchbase_data()

    # Data Cleaning and Analysis
    df = clean_and_analyze_data(df)

    # Feature Engineering
    df = engineer_features(df)

    # Model Development
    model = train_outsourcing_prediction_model(df)

    # Save the model
    with open("outsourcing_prediction_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Deployment and Ongoing Monitoring
    monitor_model_performance(model)

if __name__ == "__main__":
    main()