import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(730)]

data = []
for date in dates:
    day_of_week = date.weekday()
    is_weekend = day_of_week >= 5
    
    base_weight = 100 + 20 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
    if is_weekend:
        base_weight *= 0.7
    
    weight = base_weight + np.random.normal(0, 15)
    weight = max(10, weight)
    
    data.append({
        'Date': date.strftime('%Y-%m-%d'),
        'Weight': round(weight, 2)
    })

df = pd.DataFrame(data)
df.to_csv('Data.csv', index=False)
print(f"Generated Data.csv with {len(df)} rows")

working_days = []
for date in dates:
    day_of_week = date.weekday()
    is_working = 1 if day_of_week < 5 else 0
    working_days.append({
        'Date': date.strftime('%Y-%m-%d'),
        'WorkingDay': is_working
    })

wd_df = pd.DataFrame(working_days)
wd_df.to_csv('WorkingDay.csv', index=False)
print(f"Generated WorkingDay.csv with {len(wd_df)} rows")

print("\nSample data:")
print(df.head(10))
