import pandas as pd
import jdatetime
from sklearn.preprocessing import LabelEncoder


def cleanse_inventory_data(file_path):
    df = pd.read_excel(file_path)
    df_clean = df.copy()

    df_clean['مقدار (اصلی)'] = pd.to_numeric(
        df_clean['مقدار (اصلی)'], errors='coerce'
    ).round().astype('Int64')

    df_clean['کد کالا'] = pd.to_numeric(
        df_clean['کد کالا'], errors='coerce'
    ).astype('Int64')

    def persian_to_gregorian(date_str):
        try:
            y, m, d = map(int, str(date_str).split('/'))
            g_date = jdatetime.date(y, m, d).togregorian()
            return pd.to_datetime(g_date)
        except Exception:
            return pd.NaT

    df_clean['تاریخ'] = df_clean['تاریخ'].apply(persian_to_gregorian)
    df_clean = df_clean.sort_values('تاریخ').reset_index(drop=True)

    cols_to_check = ['نوع سند', 'وضعیت', 'ماهیت کالا']
    for col in cols_to_check:
        if col in df_clean.columns:
            unique_count = df_clean[col].nunique()
            print(
                f"Column '{col}' has {unique_count} unique values: {df_clean[col].dropna().unique()}")
            if unique_count <= 2:
                df_clean = df_clean.drop(columns=[col])
                print(f"--> Dropped low-variance column: '{col}'\n")

    if 'نام کالا' in df_clean.columns:
        df_clean = df_clean.drop(columns=['نام کالا'])
        print("Dropped redundant column: 'نام کالا' (kept 'کد کالا' as the primary numeric identifier)\n")

    categorical_features = ['واحد سنجش', 'انبار', 'طرف مقابل 4', 'محل مصرف']
    existing_cat_features = [
        col for col in categorical_features if col in df_clean.columns]

    label_encoder = LabelEncoder()
    for col in existing_cat_features:
        df_clean[col] = df_clean[col].fillna('Unknown')
        df_clean[col] = label_encoder.fit_transform(df_clean[col].astype(str))
        print(f"Encoded '{col}' into integer categories.")

    print("\nData cleansing and encoding completed successfully.")
    return df_clean


file_path = 'Data.xlsx'
df_processed = cleanse_inventory_data(file_path)
print(df_processed.head())
print(df_processed.info())
