# AI-Based Skincare Recommendation System
# Simple Machine Learning Project using Python

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Dataset Creation
# -----------------------------

data = {
    'Skin_Type': [
        'Oily', 'Dry', 'Sensitive', 'Combination',
        'Oily', 'Dry', 'Sensitive', 'Combination'
    ],
    
    'Concern': [
        'Acne', 'Dryness', 'Redness', 'Pimples',
        'Dark Spots', 'Flaky Skin', 'Irritation', 'Acne'
    ],
    
    'Recommendation': [
        'Salicylic Acid Cleanser',
        'Hyaluronic Acid Moisturizer',
        'Aloe Vera Gel',
        'Niacinamide Serum',
        'Vitamin C Serum',
        'Ceramide Moisturizer',
        'Gentle Face Wash',
        'Tea Tree Face Wash'
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# Encoding Text Data
# -----------------------------

le_skin = LabelEncoder()
le_concern = LabelEncoder()
le_recommend = LabelEncoder()

df['Skin_Type'] = le_skin.fit_transform(df['Skin_Type'])
df['Concern'] = le_concern.fit_transform(df['Concern'])
df['Recommendation'] = le_recommend.fit_transform(df['Recommendation'])

# -----------------------------
# Features and Labels
# -----------------------------

X = df[['Skin_Type', 'Concern']]
y = df['Recommendation']

# -----------------------------
# Model Training
# -----------------------------

model = DecisionTreeClassifier()
model.fit(X, y)

# -----------------------------
# User Input
# -----------------------------

print("===== AI Skincare Recommendation System =====")

skin = input("Enter your skin type (Oily/Dry/Sensitive/Combination): ")
concern = input("Enter your skin concern: ")

# Convert input into encoded values
skin_encoded = le_skin.transform([skin])[0]
concern_encoded = le_concern.transform([concern])[0]

# Prediction
prediction = model.predict([[skin_encoded, concern_encoded]])

# Decode result
recommended_product = le_recommend.inverse_transform(prediction)

# -----------------------------
# Output
# -----------------------------

print("\nRecommended Skincare Product:")
print(recommended_product[0])

print("\nThank you for using the AI Skincare Recommendation System!")
