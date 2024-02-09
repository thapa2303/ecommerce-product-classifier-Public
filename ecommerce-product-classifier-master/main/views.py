from django.shortcuts import render
from joblib import load
import csv

import numpy as np

from django.conf import settings

# text_clf = load("/home/badu/workspace/collegeProject/ecommerce-product-classifier/classification_model1.joblib")
# encoder = load("/home/badu/workspace/collegeProject/ecommerce-product-classifier/category_encoder1.joblib")
text_clf = load( settings.BASE_DIR / "classification_model_sdg1.joblib")
encoder = load( settings.BASE_DIR / "category_encoder2.joblib")

brand_clf = load(settings.BASE_DIR / "gadgets_brand_model.joblib")
brand_encoder = load(settings.BASE_DIR / "gadgets_brand_encoder.joblib")


categories_prices = {}

def load_data():
    with open(settings.BASE_DIR / "price_data.csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            try:
                category = str(row[1])
                price = float(row[2])
            except:
                continue

            try:
                categories_prices[category].append(price)
            except KeyError:
                categories_prices[category] = [price]

load_data()

def home(request):
    name = request.GET.get("name", None)
    description = request.GET.get("description", None)

    if name and description:
        classify_text = name + description
    elif name:
        classify_text = name
    elif description:
        classify_text = description
    else:
        return render(request , "home.html", {})

    categories = encoder.inverse_transform([text_clf.predict([classify_text])])
    brands = None
    if categories:
        category = categories[0][0]
        if "Electronic Accessories" == category and name:
            brands = brand_encoder.inverse_transform([brand_clf.predict([name])])
            if brands:
                brands = brands[0]
        category_prices = np.array(categories_prices[category])
        quantiles = {}
        quantiles["first"] = np.quantile(category_prices, 0.20)
        quantiles["second"] = np.quantile(category_prices, 0.40)
        quantiles["third"] = np.quantile(category_prices, 0.60)
        quantiles["fourth"] = np.quantile(category_prices, 0.80)
        quantiles["fifth"] = np.quantile(category_prices, 1.00)

        given_price = request.GET.get("price", None)
        if given_price:
            given_price = float(given_price)
            if given_price <= quantiles["first"]:
                quantiles["this"] = "Very Low"
            elif given_price <= quantiles["second"]:
                quantiles["this"] = "Low"
            elif given_price <= quantiles["third"]:
                quantiles["this"] = "Medium"
            elif given_price <= quantiles["fourth"]:
                quantiles["this"] = "High"
            else:
                quantiles["this"] = "Very High"

    return render(request , "home.html",  {"categories": categories, "brand": brands, "quantiles": quantiles})