import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Locations in Tamil Nadu
locations = [
    'Trichy', 'Chennai', 'Thiruvannamalai', 'Cuddalore', 'Karur',
    'Pondy', 'Salem', 'Erode', 'Namakkal', 'Vilupuram', 'Vellore'
]

# Generate 1000 rows
data = []
for _ in range(1000):
    square_feet = random.randint(400, 3500)  # in sqft
    bedrooms = random.randint(1, 5)
    location = random.choice(locations)
    age = random.randint(0, 40)  # years

    # Pricing logic
    base_price = 8000000  # ₹8 million
    location_factor = locations.index(location) * 100000  # location premium
    size_factor = square_feet * 250                     # more sqft = higher price
    age_discount = (40 - age) * 3000                    # newer = more expensive
    noise = random.randint(-100000, 100000)             # market randomness

    # Final price
    price = base_price + location_factor + size_factor + age_discount + noise

    data.append({
        'Square_Feet': square_feet,
        'Bedrooms': bedrooms,
        'Location': location,
        'Age_of_Property': age,
        'Price': int(price)
    })

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv('real_estate_tamilnadu.csv', index=False)

print("✅ Dataset 'real_estate_tamilnadu.csv' generated with 1000 rows.")
