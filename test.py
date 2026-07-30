import pandas as pd
import jdatetime
import numpy as np
from sklearn.preprocessing import LabelEncoder

INPUT_FILE = "Data.xlsx"
OUTPUT_FILE = "ABC_Analysis.xlsx"

ORIGINAL_DATE_COL = "تاریخ"
PERSIAN_DATE_COL = "تاریخ شمسی"
QTY_COL = "مقدار (اصلی)"
PRODUCT_CODE_COL = "کد کالا"

RENAME_COLS = {
    "طرف مقابل 4": "طرف مقابل"
}

CATEGORICAL_FEATURES = [
    "نوع سند",
    "وضعیت",
    "ماهیت کالا",
    "واحد سنجش",
    "انبار",
    "طرف مقابل",
    "محل مصرف"
]

TEXT_FEATURES = [
    "نام کالا"
]

A_THRESHOLD = 70
B_THRESHOLD = 90

PERSIAN_MONTH_NAMES = {
    1: "فروردین", 2: "اردیبهشت", 3: "خرداد", 4: "تیر",
    5: "مرداد", 6: "شهریور", 7: "مهر", 8: "آبان",
    9: "آذر", 10: "دی", 11: "بهمن", 12: "اسفند"
}


def convert_persian_digits(text):
    if not isinstance(text, str):
        return text
    for persian_digit, english_digit in zip("۰۱۲۳۴۵۶۷۸۹", "0123456789"):
        text = text.replace(persian_digit, english_digit)
    for arabic_digit, english_digit in zip("٠١٢٣٤٥٦٧٨٩", "0123456789"):
        text = text.replace(arabic_digit, english_digit)
    return text


def normalize_numeric_series(series):
    if series.dtype == object:
        s = series.astype(str).apply(convert_persian_digits)
        s = s.str.replace(",", "", regex=False)
        s = s.str.strip()
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def parse_persian_date(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    s = convert_persian_digits(s)
    s = s.split(" ")[0]
    for sep in ["/", "-", "."]:
        if sep in s:
            parts = s.split(sep)
            if len(parts) == 3:
                try:
                    year, month, day = map(int, parts)
                    return jdatetime.date(year, month, day)
                except Exception:
                    return None
    return None


def clean_base_data(df):
    print("\nStep 1: Cleaning base data...")
    df = df.copy()
    df.rename(columns=RENAME_COLS, inplace=True)
    print("Renamed specific columns.")

    if ORIGINAL_DATE_COL in df.columns:
        df.rename(columns={ORIGINAL_DATE_COL: PERSIAN_DATE_COL}, inplace=True)
        print("Renamed original Persian date column.")

    if PERSIAN_DATE_COL not in df.columns:
        raise ValueError("Date column not found.")
    if PRODUCT_CODE_COL not in df.columns:
        raise ValueError("Product code column not found.")
    if QTY_COL not in df.columns:
        raise ValueError("Quantity column not found.")

    df[PRODUCT_CODE_COL] = normalize_numeric_series(df[PRODUCT_CODE_COL])
    df[QTY_COL] = normalize_numeric_series(df[QTY_COL]).fillna(0)

    df = df.dropna(subset=[PRODUCT_CODE_COL])
    df[PRODUCT_CODE_COL] = df[PRODUCT_CODE_COL].astype("Int64")

    if (df[QTY_COL] % 1 == 0).all():
        df[QTY_COL] = df[QTY_COL].astype("Int64")

    for col in TEXT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("نامشخص").astype(str).str.strip()
    print("Cleaned text columns.")

    label_encoder = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            df[col] = label_encoder.fit_transform(df[col])
            print(f"Encoded '{col}' into integer categories.")

    print("Base data cleaning and encoding completed.")
    return df


def add_date_features(df):
    print("\nStep 2: Adding date features...")
    df = df.copy()

    df[PERSIAN_DATE_COL] = (
        df[PERSIAN_DATE_COL]
        .fillna("")
        .astype(str)
        .str.strip()
        .apply(convert_persian_digits)
    )

    parsed_dates = df[PERSIAN_DATE_COL].apply(parse_persian_date)

    df["تاریخ میلادی"] = parsed_dates.apply(
        lambda x: pd.to_datetime(x.togregorian()) if x is not None else pd.NaT
    )

    df["سال شمسی"] = parsed_dates.apply(
        lambda x: x.year if x is not None else np.nan
    ).astype("Int64")

    df["ماه شمسی"] = parsed_dates.apply(
        lambda x: f"{x.year:04d}/{x.month:02d}" if x is not None else None
    )

    df["نام ماه شمسی"] = parsed_dates.apply(
        lambda x: PERSIAN_MONTH_NAMES.get(x.month) if x is not None else None
    )

    df["روز شمسی"] = parsed_dates.apply(
        lambda x: x.day if x is not None else np.nan
    ).astype("Int64")

    df["فصل شمسی"] = parsed_dates.apply(
        lambda x: int((x.month - 1) // 3) + 1 if x is not None else np.nan
    ).astype("Int64")

    df["سال میلادی"] = df["تاریخ میلادی"].dt.year.astype("Int64")
    df["ماه میلادی"] = df["تاریخ میلادی"].dt.month.astype("Int64")
    df["روز میلادی"] = df["تاریخ میلادی"].dt.day.astype("Int64")
    df["روز هفته میلادی"] = df["تاریخ میلادی"].dt.dayofweek.astype("Int64")

    print("Added Persian and Gregorian date features.")
    return df


def add_monthly_abc_analysis(df):
    print("\nStep 3: Performing ABC analysis per Persian month...")
    abc_columns = [
        "کلاس ABC", "رتبه کالا در ماه", "مجموع مصرف کالا در ماه",
        "تعداد تراکنش کالا در ماه", "سهم درصد کالا از مصرف ماه", "درصد تجمعی ماه"
    ]

    valid_df = df.dropna(subset=["ماه شمسی"])

    if valid_df.empty:
        for col in abc_columns:
            df[col] = None
        return df

    monthly = valid_df.groupby(
        ["ماه شمسی", PRODUCT_CODE_COL], as_index=False
    ).agg({QTY_COL: ["sum", "size"]})

    monthly.columns = [
        "ماه شمسی", PRODUCT_CODE_COL,
        "مجموع مصرف کالا در ماه", "تعداد تراکنش کالا در ماه"
    ]

    abc_frames = []

    for month, group in monthly.groupby("ماه شمسی", sort=True):
        group = group.sort_values(
            "مجموع مصرف کالا در ماه", ascending=False).copy()
        total_qty = group["مجموع مصرف کالا در ماه"].sum()
        group["رتبه کالا در ماه"] = range(1, len(group) + 1)

        if total_qty > 0:
            share_pct = (group["مجموع مصرف کالا در ماه"] / total_qty) * 100
            cumulative_pct = share_pct.cumsum()
            prev_cumulative_pct = cumulative_pct.shift(1).fillna(0)

            conditions = [
                prev_cumulative_pct < A_THRESHOLD,
                (prev_cumulative_pct >= A_THRESHOLD) & (
                    prev_cumulative_pct < B_THRESHOLD),
                prev_cumulative_pct >= B_THRESHOLD
            ]
            choices = ["A", "B", "C"]
            group["کلاس ABC"] = np.select(conditions, choices, default="C")
        else:
            share_pct = pd.Series(0.0, index=group.index)
            cumulative_pct = pd.Series(0.0, index=group.index)
            group["کلاس ABC"] = "C"

        group["سهم درصد کالا از مصرف ماه"] = share_pct.round(2)
        group["درصد تجمعی ماه"] = cumulative_pct.round(2)
        abc_frames.append(group)

    monthly_abc = pd.concat(abc_frames, ignore_index=True)

    df = df.merge(monthly_abc, on=["ماه شمسی", PRODUCT_CODE_COL], how="left")
    print("Calculated and merged ABC classification.")
    return df


def reorder_columns(df):
    print("\nStep 4: Reordering columns...")
    base_cols = [
        PERSIAN_DATE_COL, "انبار", PRODUCT_CODE_COL, "نام کالا", "واحد سنجش",
        QTY_COL, "طرف مقابل", "محل مصرف", "نوع سند", "وضعیت", "ماهیت کالا"
    ]

    date_cols = [
        "سال شمسی", "ماه شمسی", "نام ماه شمسی", "روز شمسی", "فصل شمسی",
        "تاریخ میلادی", "سال میلادی", "ماه میلادی", "روز میلادی", "روز هفته میلادی"
    ]

    abc_cols = [
        "کلاس ABC", "رتبه کالا در ماه", "مجموع مصرف کالا در ماه",
        "تعداد تراکنش کالا در ماه", "سهم درصد کالا از مصرف ماه", "درصد تجمعی ماه"
    ]

    preferred_cols = base_cols + date_cols + abc_cols
    ordered_cols = [col for col in preferred_cols if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in ordered_cols]

    print("Columns reordered successfully.")
    return df[ordered_cols + remaining_cols]


def main():
    print(f"Loading data from {INPUT_FILE}...")
    raw_df = pd.read_excel(INPUT_FILE)
    raw_df.columns = [str(col).strip() for col in raw_df.columns]
    initial_rows = len(raw_df)
    print(f"Loaded {initial_rows} initial rows.")

    df = clean_base_data(raw_df)
    rows_after_product_cleaning = len(df)

    df = add_date_features(df)

    rows_before_date_drop = len(df)
    df = df.dropna(subset=["تاریخ میلادی", "ماه شمسی"]).reset_index(drop=True)
    rows_after_date_drop = len(df)
    print(f"Dropped invalid dates. Rows remaining: {rows_after_date_drop}")

    df = add_monthly_abc_analysis(df)

    df = df.sort_values(
        by=["تاریخ میلادی", PRODUCT_CODE_COL, QTY_COL],
        ascending=[True, True, False]
    ).reset_index(drop=True)
    print("Sorted data by date, product code, and quantity.")

    df = reorder_columns(df)

    print(f"Exporting data to {OUTPUT_FILE}...")
    df.to_excel(OUTPUT_FILE, index=False, sheet_name="Data")
    print("Export completed successfully.")

    print(f"\nInitial rows: {initial_rows}")

    if rows_after_product_cleaning < initial_rows:
        print(
            f"Rows removed due to missing product code: {initial_rows - rows_after_product_cleaning}")
    if rows_after_date_drop < rows_before_date_drop:
        print(
            f"Rows removed due to invalid or missing date: {rows_before_date_drop - rows_after_date_drop}")

    print(f"Final shape: {df.shape}")
    print("\nABC class counts:")
    print(df["کلاس ABC"].value_counts(dropna=False))
    print("\nFinal columns:")
    print(list(df.columns))


if __name__ == "__main__":
    main()
