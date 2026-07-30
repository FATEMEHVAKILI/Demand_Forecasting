import os
import warnings
import numpy as np
import pandas as pd
import jdatetime
from datetime import timedelta
from sklearn .preprocessing import LabelEncoder, MinMaxScaler
from sklearn .ensemble import RandomForestRegressor
from sklearn .model_selection import GridSearchCV
from sklearn .feature_selection import mutual_info_regression
from sklearn .metrics import mean_absolute_error, mean_squared_error
warnings .filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception as e:
    print(f"XGBoost not available: {e}")
    XGB_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception as e:
    print(f"Prophet not available: {e}")
    PROPHET_AVAILABLE = False

try:
    from statsmodels .tsa .statespace .sarimax import SARIMAX
    SARIMA_AVAILABLE = True
except Exception as e:
    print(f"SARIMA not available: {e}")
    SARIMA_AVAILABLE = False

try:
    import torch
    import torch .nn as nn
    import torch .optim as optim
    from torch .utils .data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False

INPUT_FILE = "Data.xlsx"
WORKING_DAY_FILE = "WorkingDay.csv"
POWERBI_OUTPUT = "PowerBI_Data.xlsx"
FORECAST_OUTPUT = "Forecast_Results.xlsx"
ABC_OUTPUT = "ABC_Analysis.xlsx"

ORIGINAL_DATE_COL = "تاریخ"
PERSIAN_DATE_COL = "تاریخ شمسی"
QTY_COL = "مقدار (اصلی)"
PRODUCT_CODE_COL = "کد کالا"
MONTH_START_COL = "تاریخ شروع ماه میلادی"

FORECAST_HORIZON = 6
A_THRESHOLD = 70
B_THRESHOLD = 90

RENAME_COLS = {"طرف مقابل 4": "طرف مقابل"}

LABEL_ENCODED_COLS = [
    "نوع سند", "وضعیت", "ماهیت کالا", "واحد سنجش",
    "انبار", "طرف مقابل", "محل مصرف"
]
TEXT_FEATURES = ["نام کالا"]

PERSIAN_MONTH_NAMES = {
    1: "فروردین", 2: "اردیبهشت", 3: "خرداد", 4: "تیر",
    5: "مرداد", 6: "شهریور", 7: "مهر", 8: "آبان",
    9: "آذر", 10: "دی", 11: "بهمن", 12: "اسفند"
}


class BiLSTMRegressor (nn .Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(BiLSTMRegressor, self).__init__()
        self .bilstm = nn .LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self .dropout = nn .Dropout(dropout)
        self .fc = nn .Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self .bilstm(x)
        lstm_out = self .dropout(lstm_out)
        return self .fc(lstm_out[:, -1, :])


def convert_persian_digits(text):
    if not isinstance(text, str):
        return text
    for p, e in zip("۰۱۲۳۴۵۶۷۸۹", "0123456789"):
        text = text .replace(p, e)
    for p, e in zip("٠١٢٣٤٥٦٧٨٩", "0123456789"):
        text = text .replace(p, e)
    return text


def normalize_numeric_series(series):
    if series .dtype == object:
        s = series .astype(str).apply(convert_persian_digits).str .replace(
            ",", "", regex=False).str .strip()
        return pd .to_numeric(s, errors="coerce")
    return pd .to_numeric(series, errors="coerce")


def parse_persian_date(value):
    if pd .isna(value):
        return None
    s = str(value).strip().split(" ")[0]
    s = convert_persian_digits(s)
    for sep in ["/", "-", "."]:
        if sep in s:
            parts = s .split(sep)
            if len(parts) == 3:
                try:
                    return jdatetime .date(int(parts[0]), int(parts[1]), int(parts[2]))
                except Exception:
                    return None
    return None


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np .asarray(
        y_true, dtype=float), np .asarray(y_pred, dtype=float)
    denominator = np .where(y_true == 0, 1.0, y_true)
    return np .mean(np .abs((y_true - y_pred)/denominator))*100


def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np .asarray(
        y_true, dtype=float), np .asarray(y_pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np .sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": calculate_mape(y_true, y_pred)
    }


def generate_persian_month_list(last_gregorian_date, horizon=6):
    last_date = pd .to_datetime(last_gregorian_date)
    persian_months = []
    current_date = (last_date + pd .DateOffset(months=1)).replace(day=1)
    for _ in range(horizon):
        jd = jdatetime .date .fromgregorian(
            year=current_date .year, month=current_date .month, day=current_date .day)
        persian_months .append(f"{jd.year}/{jd.month:02d}")
        current_date = current_date + pd .DateOffset(months=1)
    return persian_months


def get_persian_features(ts):
    jd = jdatetime .date .fromgregorian(
        year=ts .year, month=ts .month, day=ts .day)
    return {
        "ماه شمسی": f"{jd.year:04d}/{jd.month:02d}",
        "نام ماه شمسی": PERSIAN_MONTH_NAMES .get(jd .month),
        "فصل شمسی": int((jd .month - 1)//3)+1
    }


def add_working_day_features(df, working_day_file):

    print("Adding working day features...")
    wd_df = pd .read_csv(working_day_file)
    wd_df['Date'] = pd .to_datetime(wd_df['Date'], format='%d/%m/%Y')
    df['تاریخ میلادی'] = pd .to_datetime(df['تاریخ میلادی'])
    wd_df = wd_df .rename(columns={'WorkingDay': 'روز کاری'})
    df = df .merge(wd_df[['Date', 'روز کاری']],
                   left_on='تاریخ میلادی', right_on='Date', how='left')
    df = df .drop(columns=['Date'])
    df['روز کاری'] = df['روز کاری'].fillna(0).astype(int)
    return df


def add_monthly_abc_analysis(df):

    print("Performing monthly ABC analysis...")
    valid_df = df .dropna(subset=["ماه شمسی"]).copy()
    if valid_df .empty:
        for col in ["کلاس ABC", "رتبه کالا در ماه", "مجموع مصرف کالا در ماه",
                    "تعداد تراکنش کالا در ماه", "سهم درصد کالا از مصرف ماه", "درصد تجمعی ماه"]:
            df[col] = None
        return df

    monthly = valid_df .groupby(
        ["ماه شمسی", PRODUCT_CODE_COL], as_index=False
    ).agg({QTY_COL: ["sum", "size"]})
    monthly .columns = [
        "ماه شمسی", PRODUCT_CODE_COL,
        "مجموع مصرف کالا در ماه", "تعداد تراکنش کالا در ماه"
    ]

    abc_frames = []
    for month, group in monthly .groupby("ماه شمسی", sort=True):
        group = group .sort_values(
            "مجموع مصرف کالا در ماه", ascending=False).copy()
        total_qty = group["مجموع مصرف کالا در ماه"].sum()
        group["رتبه کالا در ماه"] = range(1, len(group)+1)

        if total_qty > 0:
            share_pct = (group["مجموع مصرف کالا در ماه"]/total_qty)*100
            cumulative_pct = share_pct .cumsum()
            prev_cumulative_pct = cumulative_pct .shift(1).fillna(0)

            conditions = [
                prev_cumulative_pct < A_THRESHOLD,
                (prev_cumulative_pct >= A_THRESHOLD) & (
                    prev_cumulative_pct < B_THRESHOLD),
                prev_cumulative_pct >= B_THRESHOLD
            ]
            choices = ["A", "B", "C"]
            group["کلاس ABC"] = np .select(conditions, choices, default="C")
        else:
            share_pct = pd .Series(0.0, index=group .index)
            cumulative_pct = pd .Series(0.0, index=group .index)
            group["کلاس ABC"] = "C"

        group["سهم درصد کالا از مصرف ماه"] = share_pct .round(2)
        group["درصد تجمعی ماه"] = cumulative_pct .round(2)
        abc_frames .append(group)

    monthly_abc = pd .concat(abc_frames, ignore_index=True)
    df = df .merge(monthly_abc, on=["ماه شمسی", PRODUCT_CODE_COL], how="left")
    return df


def winsorize_outliers(df):

    df = df .copy()
    df["log_qty"] = np .log1p(df[QTY_COL].clip(lower=0).astype(float))
    q1 = df .groupby(PRODUCT_CODE_COL)["log_qty"].transform(
        lambda x: x .quantile(0.25)if len(x) >= 10 else np .nan
    ).fillna(df["log_qty"].quantile(0.25))
    q3 = df .groupby(PRODUCT_CODE_COL)["log_qty"].transform(
        lambda x: x .quantile(0.75)if len(x) >= 10 else np .nan
    ).fillna(df["log_qty"].quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df["original_qty"] = df[QTY_COL]
    df[QTY_COL] = np .expm1(np .clip(df["log_qty"], lower, upper)).round().clip(
        lower=0).astype("Int64")
    return df .drop(columns=["log_qty"])


def clean_base_data(df):

    df = df .copy().rename(columns=RENAME_COLS)
    if ORIGINAL_DATE_COL in df .columns:
        df .rename(columns={ORIGINAL_DATE_COL: PERSIAN_DATE_COL}, inplace=True)
    if PERSIAN_DATE_COL not in df .columns:
        raise ValueError("Date column not found.")

    df[PRODUCT_CODE_COL] = normalize_numeric_series(
        df[PRODUCT_CODE_COL]).astype("Int64")
    df[QTY_COL] = normalize_numeric_series(df[QTY_COL]).fillna(0)
    if (df[QTY_COL] % 1 == 0).all():
        df[QTY_COL] = df[QTY_COL].astype("Int64")

    for col in TEXT_FEATURES:
        if col in df .columns:
            df[col] = df[col].fillna("نامشخص").astype(str).str .strip()

    label_encoder = LabelEncoder()
    for col in LABEL_ENCODED_COLS:
        if col in df .columns:
            df[col] = label_encoder .fit_transform(
                df[col].fillna("Unknown").astype(str).str .strip())
    return df


def add_full_date_features(df):

    df = df .copy()
    df[PERSIAN_DATE_COL] = df[PERSIAN_DATE_COL].astype(
        str).str .strip().apply(convert_persian_digits)
    parsed = df[PERSIAN_DATE_COL].apply(parse_persian_date)

    df["تاریخ میلادی"] = parsed .apply(
        lambda x: pd .to_datetime(x .togregorian())if x else pd .NaT)
    df["سال شمسی"] = parsed .apply(
        lambda x: x .year if x else np .nan).astype("Int64")
    df["ماه شمسی"] = parsed .apply(
        lambda x: f"{x.year:04d}/{x.month:02d}"if x else None)
    df["نام ماه شمسی"] = parsed .apply(
        lambda x: PERSIAN_MONTH_NAMES .get(x .month)if x else None)
    df["روز شمسی"] = parsed .apply(
        lambda x: x .day if x else np .nan).astype("Int64")
    df["فصل شمسی"] = parsed .apply(lambda x: int(
        (x .month - 1)//3)+1 if x else np .nan).astype("Int64")

    df["سال میلادی"] = df["تاریخ میلادی"].dt .year .astype("Int64")
    df["ماه میلادی"] = df["تاریخ میلادی"].dt .month .astype("Int64")
    df["روز میلادی"] = df["تاریخ میلادی"].dt .day .astype("Int64")
    df["روز هفته میلادی"] = df["تاریخ میلادی"].dt .dayofweek .astype("Int64")

    df[MONTH_START_COL] = df["تاریخ میلادی"].dt .to_period(
        "M").dt .to_timestamp()
    return df


def build_monthly_product_panel(df):

    valid = df .dropna(subset=[MONTH_START_COL]).copy()
    months = pd .date_range(valid[MONTH_START_COL].min(),
                            valid[MONTH_START_COL].max(), freq="MS")
    products = valid[[PRODUCT_CODE_COL, "نام کالا"]].drop_duplicates()
    base = products .merge(pd .DataFrame(
        {MONTH_START_COL: months}), how="cross")

    agg_dict = {
        QTY_COL: "sum",
        "original_qty": "sum"
    }

    optional_aggs = {
        "روز کاری": "sum",
        "تعداد تراکنش کالا در ماه": "max",
        "سهم درصد کالا از مصرف ماه": "max",
        "درصد تجمعی ماه": "max",
        "رتبه کالا در ماه": "max",
        "سال شمسی": "max",
        "فصل شمسی": "max"
    }

    for col, func in optional_aggs .items():
        if col in valid .columns:
            agg_dict[col] = func

    agg = valid .groupby([MONTH_START_COL, PRODUCT_CODE_COL],
                         as_index=False).agg(agg_dict)

    agg = agg .rename(columns={QTY_COL: "target",
                               "original_qty": "original_target"})

    panel = base .merge(
        agg, on=[MONTH_START_COL, PRODUCT_CODE_COL], how="left")
    panel[["target", "original_target"]] = panel[[
        "target", "original_target"]].fillna(0)

    for col in agg_dict .keys():
        if col not in [QTY_COL, "original_qty"]:
            panel[col] = panel[col].fillna(0)

    panel["ماه شمسی"] = panel[MONTH_START_COL].apply(
        lambda x: get_persian_features(x)["ماه شمسی"]
    )

    abc_info = valid[["ماه شمسی", PRODUCT_CODE_COL,
                      "کلاس ABC"]].drop_duplicates()
    panel = panel .merge(
        abc_info, on=[PRODUCT_CODE_COL, "ماه شمسی"], how="left")
    panel["کلاس ABC"] = panel["کلاس ABC"].fillna("C")
    panel["abc_encoded"] = panel["کلاس ABC"].map(
        {"A": 1, "B": 2, "C": 3}).astype(int)

    panel["ماه میلادی"] = panel[MONTH_START_COL].dt .month
    panel["فصل شمسی"] = panel[MONTH_START_COL].apply(
        lambda x: get_persian_features(x)["فصل شمسی"]
    )

    panel = panel .sort_values(
        [PRODUCT_CODE_COL, MONTH_START_COL]).reset_index(drop=True)

    for lag in [1, 2, 3]:
        panel[f"lag_{lag}"] = panel .groupby(PRODUCT_CODE_COL)[
            "target"].shift(lag).fillna(0)

    panel["rolling_mean_3"] = panel .groupby(PRODUCT_CODE_COL)["target"].transform(
        lambda x: x .shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

    panel["rolling_std_3"] = panel .groupby(PRODUCT_CODE_COL)["target"].transform(
        lambda x: x .shift(1).rolling(3, min_periods=1).std()
    ).fillna(0)

    return panel


def prepare_features_for_models(panel):

    exclude_cols = [
        PRODUCT_CODE_COL,
        "نام کالا",
        MONTH_START_COL,
        "ماه شمسی",
        "target",
        "original_target",
        "کلاس ABC"
    ]

    numeric_cols = panel .select_dtypes(include=['number']).columns .tolist()

    feature_candidates = [
        col for col in numeric_cols if col not in exclude_cols]

    for col in feature_candidates:
        panel[col] = pd .to_numeric(
            panel[col], errors='coerce').fillna(0).astype(float)

    print(
        f"\nDynamically identified {len(feature_candidates)} feature candidates for importance analysis:")
    for col in feature_candidates:
        print(f"   • {col}")

    return panel, feature_candidates


def analyze_feature_importance(panel, feature_candidates, target_col="target"):

    print("\nFEATURE IMPORTANCE ANALYSIS")

    data = panel .dropna(subset=[target_col]+feature_candidates).copy()
    X = data[feature_candidates].fillna(0)
    y = data[target_col]

    corr_importance = []
    for col in feature_candidates:
        try:
            corr = np .abs(X[col].astype(float).corr(y .astype(float)))
            corr_importance .append(corr if not np .isnan(corr)else 0.0)
        except Exception:
            corr_importance .append(0.0)

    mi_importance = mutual_info_regression(X, y, random_state=123)

    rf = RandomForestRegressor(n_estimators=100, random_state=123, n_jobs=-1)
    rf .fit(X, y)
    rf_importance = rf .feature_importances_

    xgb_importance = np .zeros(len(feature_candidates))
    if XGB_AVAILABLE:
        xgb = XGBRegressor(objective="reg:squarederror",
                           random_state=123, n_jobs=1, verbosity=0)
        xgb .fit(X, y)
        xgb_importance = xgb .feature_importances_

    importance_df = pd .DataFrame({
        "Feature": feature_candidates,
        "Correlation_Abs": np .round(corr_importance, 4),
        "Mutual_Information": np .round(mi_importance, 4),
        "RF_Importance": np .round(rf_importance, 4),
        "XGB_Importance": np .round(xgb_importance, 4)
    }).sort_values("Mutual_Information", ascending=False).reset_index(drop=True)

    print("\nFull Feature Importance Table (Sorted by Mutual Information):")
    print(importance_df .to_string(index=False))

    top_n = 10
    selected_features = importance_df .head(top_n)["Feature"].tolist()

    return importance_df, selected_features


def train_selected_models(X_train, y_train, total, models_to_train=["all"], horizon=6):
    print(f"\nTraining selected models: {models_to_train}")
    models = {}
    eval_results = []
    can_grid = len(X_train) >= 10

    if "all" in models_to_train or "RF" in models_to_train:
        rf = RandomForestRegressor(random_state=123, n_jobs=-1)
        param_grid_rf = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_leaf': [1, 3, 5],
            'min_samples_split': [2, 5]
        }
        if can_grid:
            rf_search = GridSearchCV(
                rf, param_grid_rf, scoring="neg_mean_absolute_error", cv=3, n_jobs=-1)
            rf_search .fit(X_train, y_train)
            models["Random Forest"] = rf_search .best_estimator_
            print(f"RF best params: {rf_search.best_params_}")
        else:
            rf .fit(X_train, y_train)
            models["Random Forest"] = rf

    if "all" in models_to_train or "XGB" in models_to_train:
        if XGB_AVAILABLE:
            xgb = XGBRegressor(objective="reg:squarederror",
                               random_state=123, n_jobs=1, verbosity=0)
            param_grid_xgb = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'min_child_weight': [1, 3]
            }
            if can_grid:
                xgb_search = GridSearchCV(
                    xgb, param_grid_xgb, scoring="neg_mean_absolute_error", cv=3, n_jobs=1)
                xgb_search .fit(X_train, y_train)
                models["XGBoost"] = xgb_search .best_estimator_
                print(f"XGB best params: {xgb_search.best_params_}")
            else:
                xgb .set_params(n_estimators=100, max_depth=5,
                                learning_rate=0.05, min_child_weight=1)
                xgb .fit(X_train, y_train)
                models["XGBoost"] = xgb
        else:
            print("Warning: XGBoost not available.")

    if "all" in models_to_train or "Prophet" in models_to_train:
        if PROPHET_AVAILABLE and len(total) >= 6:
            best_prophet_model, best_mae, best_params = None, float('inf'), {}
            train, test = total .iloc[:-2].copy(), total .iloc[-2:].copy()
            train_df = train .rename(columns={MONTH_START_COL: "ds"})[
                ["ds", "y"]]

            for cps in [0.01, 0.05, 0.1]:
                for mode in ['additive', 'multiplicative']:
                    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                                daily_seasonality=False, changepoint_prior_scale=cps, seasonality_mode=mode)
                    m .fit(train_df)
                    pred = m .predict(m .make_future_dataframe(
                        periods=len(test), freq="MS")).tail(len(test))
                    mae = mean_absolute_error(
                        test['y'].values, pred['yhat'].values)
                    if mae < best_mae:
                        best_mae, best_prophet_model, best_params = mae, m, {
                            'cps': cps, 'mode': mode}
            print(
                f"Prophet best params: {best_params}, Test MAE: {best_mae:.4f}")
            models["Prophet"] = best_prophet_model
            eval_results .append({"Model": "Prophet", "MAE": best_mae})

    if "all" in models_to_train or "SARIMA" in models_to_train:
        if SARIMA_AVAILABLE and len(total) >= 6:
            train, test = total .iloc[:-2]["y"].astype(
                float), total .iloc[-2:]["y"].astype(float)
            best_sarima_model, best_mae, best_order = None, float('inf'), None

            orders = [(1, 0, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)]
            seasonal_orders = [(0, 0, 0, 12), (1, 1, 1, 12), (0, 1, 1, 12)]

            for order in orders:
                for s_order in seasonal_orders:
                    try:
                        m = SARIMAX(train, order=order, seasonal_order=s_order,
                                    enforce_stationarity=False, enforce_invertibility=False)

                        fitted = m .fit(disp=0, maxiter=50)
                        pred = fitted .forecast(steps=len(test))
                        mae = mean_absolute_error(test .values, pred .values)
                        if mae < best_mae:
                            best_mae, best_sarima_model, best_order = mae, fitted, (
                                order, s_order)
                    except Exception:
                        continue
            print(f"SARIMA best order: {best_order}, Test MAE: {best_mae:.4f}")
            models["SARIMA"] = best_sarima_model
            eval_results .append({"Model": "SARIMA", "MAE": best_mae})

    return models, pd .DataFrame(eval_results)


def forecast_ml_products(panel, model, model_name, selected_features, horizon):
    if model is None:
        return pd .DataFrame()
    future_months = pd .date_range(panel[MONTH_START_COL].max()+pd .DateOffset(months=1),
                                   periods=horizon, freq="MS")
    records = []
    for product_code, group in panel .groupby(PRODUCT_CODE_COL):
        if group["original_target"].sum() <= 0:
            continue
        group = group .sort_values(MONTH_START_COL)
        targets = group["target"].astype(float).tolist()
        abc_encoded = int(group["abc_encoded"].iloc[-1])
        product_name = group["نام کالا"].iloc[-1]

        for horizon_step, month_start in enumerate(future_months, start=1):
            last_values = targets[-3:]if len(targets) >= 3 else targets
            feature_row = pd .DataFrame([{
                "ماه میلادی": month_start .month,
                "فصل شمسی": get_persian_features(month_start)["فصل شمسی"],
                "abc_encoded": abc_encoded,
                "lag_1": targets[-1]if len(targets) >= 1 else 0.0,
                "lag_2": targets[-2]if len(targets) >= 2 else 0.0,
                "lag_3": targets[-3]if len(targets) >= 3 else 0.0,
                "rolling_mean_3": float(np .mean(last_values))if len(last_values) > 0 else 0.0,
                "rolling_std_3": float(np .std(last_values, ddof=0))if len(last_values) > 1 else 0.0
            }])[selected_features].fillna(0)

            predicted = max(0.0, float(model .predict(feature_row)[0]))
            targets .append(predicted)
            persian = get_persian_features(month_start)
            records .append({
                PRODUCT_CODE_COL: product_code,
                "نام کالا": product_name,
                "ds": month_start,
                "ماه شمسی": persian["ماه شمسی"],
                "horizon": horizon_step,
                "forecast": predicted,
                "Model": model_name
            })
    return pd .DataFrame(records)


def forecast_bilstm_products(panel, selected_features, horizon):
    if not TORCH_AVAILABLE:
        return pd .DataFrame(), pd .DataFrame()
    records, lstm_metrics = [], []
    future_months = pd .date_range(panel[MONTH_START_COL].max()+pd .DateOffset(months=1),
                                   periods=horizon, freq="MS")

    for product_code in panel[PRODUCT_CODE_COL].unique():
        product_data = panel[panel[PRODUCT_CODE_COL] ==
                             product_code].sort_values(MONTH_START_COL)
        if product_data["original_target"].sum() <= 0 or len(product_data) < 6:
            continue
        X = product_data[selected_features].fillna(0).values
        y = product_data["target"].values
        split_idx = int(len(X)*0.8)
        if split_idx < 4 or len(X)-split_idx < 2:
            continue

        model, scaler, y_scaler, test_metrics = train_bilstm(
            X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:],
            input_size=X .shape[1], epochs=10)
        if model is None:
            continue
        if test_metrics:
            lstm_metrics .append({"Model": "Bi-LSTM", **test_metrics})

        last_features = X[-1:].copy()
        recent_targets = list(product_data["target"].values[-3:])
        device = next(model .parameters()).device
        product_name = product_data["نام کالا"].iloc[-1]

        for horizon_step, month_start in enumerate(future_months, start=1):
            last_scaled = scaler .transform(last_features)
            with torch .no_grad():
                pred = model(torch .FloatTensor(
                    last_scaled .reshape(1, 1, last_scaled .shape[1])).to(device)).cpu().numpy()
                predicted = max(0.0, y_scaler .inverse_transform(pred)[0, 0])

            persian = get_persian_features(month_start)
            records .append({
                PRODUCT_CODE_COL: product_code,
                "نام کالا": product_name,
                "ds": month_start,
                "ماه شمسی": persian["ماه شمسی"],
                "horizon": horizon_step,
                "forecast": predicted,
                "Model": "Bi-LSTM"
            })
            recent_targets .append(predicted)
            if len(recent_targets) > 3:
                recent_targets .pop(0)

            if "lag_1" in selected_features:
                last_features[0][selected_features .index("lag_1")] = predicted
            if "lag_2" in selected_features:
                last_features[0][selected_features .index(
                    "lag_2")] = last_features[0][selected_features .index("lag_1")]
            if "rolling_mean_3" in selected_features:
                last_features[0][selected_features .index(
                    "rolling_mean_3")] = float(np .mean(recent_targets))

    return pd .DataFrame(records), pd .DataFrame(lstm_metrics)if lstm_metrics else pd .DataFrame()


def train_bilstm(X_train, y_train, X_test, y_test, input_size, epochs=20):
    if not TORCH_AVAILABLE:
        return None, None, None, None

    device = torch .device("cuda"if torch .cuda .is_available()else "cpu")

    param_grid = [
        {'hidden_size': 32, 'num_layers': 1, 'dropout': 0.1,
         'learning_rate': 0.01, 'batch_size': 32},
        {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2,
         'learning_rate': 0.005, 'batch_size': 32},
        {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3,
         'learning_rate': 0.001, 'batch_size': 64}
    ]

    best_model, best_scaler, best_y_scaler, best_mae = None, None, None, float(
        'inf')

    for params in param_grid:
        scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
        X_train_scaled = scaler .fit_transform(X_train)
        X_test_scaled = scaler .transform(X_test)
        y_train_scaled = y_scaler .fit_transform(y_train .reshape(-1, 1))
        y_test_scaled = y_scaler .transform(y_test .reshape(-1, 1))

        X_train_tensor = torch .FloatTensor(X_train_scaled .reshape(
            X_train_scaled .shape[0], 1, X_train_scaled .shape[1])).to(device)
        y_train_tensor = torch .FloatTensor(y_train_scaled).to(device)
        X_test_tensor = torch .FloatTensor(X_test_scaled .reshape(
            X_test_scaled .shape[0], 1, X_test_scaled .shape[1])).to(device)
        y_test_tensor = torch .FloatTensor(y_test_scaled).to(device)

        model = BiLSTMRegressor(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(device)

        criterion = nn .L1Loss()
        optimizer = optim .Adam(model .parameters(),
                                lr=params['learning_rate'])
        train_loader = DataLoader(TensorDataset(
            X_train_tensor, y_train_tensor), batch_size=params['batch_size'], shuffle=True)

        model .train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer .zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss .backward()
                optimizer .step()

        model .eval()
        with torch .no_grad():
            y_test_pred = y_scaler .inverse_transform(
                model(X_test_tensor).cpu().numpy())
            y_test_actual = y_scaler .inverse_transform(y_test_scaled)

        current_mae = mean_absolute_error(
            y_test_actual .flatten(), y_test_pred .flatten())

        if current_mae < best_mae:
            best_mae = current_mae
            best_model = model
            best_scaler = scaler
            best_y_scaler = y_scaler

    best_model .eval()
    with torch .no_grad():
        final_pred = best_y_scaler .inverse_transform(
            best_model(X_test_tensor).cpu().numpy())
        final_actual = best_y_scaler .inverse_transform(y_test_scaled)

    metrics = calculate_metrics(final_actual .flatten(), final_pred .flatten())
    return best_model, best_scaler, best_y_scaler, metrics


def format_forecast_output(df, last_date, horizon=6, is_total=True):
    if df is None or (isinstance(df, pd .DataFrame) and df .empty):
        return pd .DataFrame()
    persian_months = generate_persian_month_list(last_date, horizon)
    res = df .copy()
    res["Months"] = list(range(1, horizon + 1))*(len(res)//horizon)
    res["Persian Month"] = persian_months * (len(res)//horizon)
    res = res .rename(columns={
        "Model": "Model Name", "forecast": "Forecasting Value", "yhat": "Forecasting Value"})
    if is_total:
        core_cols = ["Months", "Persian Month",
                     "Model Name", "Forecasting Value"]
    else:
        core_cols = [PRODUCT_CODE_COL, "نام کالا", "Months",
                     "Persian Month", "Model Name", "Forecasting Value"]
    extra_cols = [c for c in res .columns if c not in core_cols]
    return res[core_cols + extra_cols]


def build_total_forecast_sheet(total, prophet_forecast, sarima_forecast,
                               ml_product_forecasts, lstm_product_forecasts, horizon=6):
    frames = []
    last_date = total[MONTH_START_COL].max()
    persian_months = generate_persian_month_list(last_date, horizon)

    historical = total .rename(
        columns={MONTH_START_COL: "ds", "y": "Forecasting Value"})
    historical["Model Name"] = "Actual"
    historical["Months"] = 0
    historical["Persian Month"] = historical["ds"].apply(
        lambda x: f"{jdatetime.date.fromgregorian(date=x).year}/{jdatetime.date.fromgregorian(date=x).month:02d}"if pd .notnull(x)else "")
    frames .append(
        historical[["Months", "Persian Month", "Model Name", "Forecasting Value"]])

    for df, name in [(prophet_forecast, "Prophet"), (sarima_forecast, "SARIMA")]:
        if df is not None and not (isinstance(df, pd .DataFrame) and df .empty):
            df_out = df .copy()
            df_out["Model Name"] = name
            df_out["Months"] = list(range(1, horizon + 1))
            df_out["Persian Month"] = persian_months
            val_col = "Forecasting Value"if "Forecasting Value" in df_out .columns else "yhat"
            frames .append(df_out .rename(columns={val_col: "Forecasting Value"})[
                ["Months", "Persian Month", "Model Name", "Forecasting Value"]])

    for ml_df, name in [(ml_product_forecasts, "ML Aggregate"), (lstm_product_forecasts, "Bi-LSTM Aggregate")]:
        if ml_df is not None and not (isinstance(ml_df, pd .DataFrame) and ml_df .empty):
            val_col = "Forecasting Value"if "Forecasting Value" in ml_df .columns else "forecast"
            total_agg = ml_df .groupby("ds", as_index=False)[val_col].sum()
            total_agg["Model Name"] = name
            total_agg["Months"] = list(range(1, horizon + 1))
            total_agg["Persian Month"] = persian_months
            frames .append(total_agg .rename(columns={val_col: "Forecasting Value"})[
                ["Months", "Persian Month", "Model Name", "Forecasting Value"]])

    return pd .concat(frames, ignore_index=True).sort_values(["Model Name", "Months"]).reset_index(drop=True)


def main(models_to_train=["XGB"]):
    print("="*60)
    print("STARTING FULL WORKFLOW")
    print("="*60)

    raw_df = pd .read_excel(INPUT_FILE)
    raw_df .columns = [str(col).strip()for col in raw_df .columns]
    print(f"Loaded raw data: {raw_df.shape}")

    df = clean_base_data(raw_df)
    df = add_full_date_features(df)
    df = add_working_day_features(df, WORKING_DAY_FILE)
    df = df .dropna(subset=["تاریخ میلادی", "ماه شمسی"]).reset_index(drop=True)

    df = df .sort_values(by=PERSIAN_DATE_COL).reset_index(drop=True)

    df = add_monthly_abc_analysis(df)

    df .to_excel(ABC_OUTPUT, index=False, sheet_name="Data")
    print(f"ABC Analysis exported to {ABC_OUTPUT}")

    df = winsorize_outliers(df)

    panel = build_monthly_product_panel(df)
    panel, feature_candidates = prepare_features_for_models(panel)
    print(f"Initial feature candidates: {feature_candidates}")

    all_months = sorted(panel[MONTH_START_COL].unique())
    split_idx = int(len(all_months)*0.8)
    train_months = all_months[:split_idx]
    test_months = all_months[split_idx:]
    train_panel = panel[panel[MONTH_START_COL].isin(train_months)].copy()
    test_panel = panel[panel[MONTH_START_COL].isin(test_months)].copy()

    X_train = train_panel[feature_candidates].fillna(0)
    y_train = train_panel["target"]
    X_test = test_panel[feature_candidates].fillna(0)
    y_test = test_panel["target"]

    importance_df, selected_features = analyze_feature_importance(
        panel, feature_candidates, target_col="target")

    print(f"\nProceeding with {len(selected_features)} selected features.")
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    total = panel .groupby(MONTH_START_COL, as_index=False)[
        "target"].sum().sort_values(MONTH_START_COL).reset_index(drop=True)
    total = total .rename(columns={"target": "y"})

    models, ml_eval = train_selected_models(
        X_train, y_train, total, models_to_train=models_to_train, horizon=FORECAST_HORIZON)

    best_ml_name = ml_eval .sort_values(
        "MAE").iloc[0]["Model"]if not ml_eval .empty else "Random Forest"
    best_ml_model = models .get(best_ml_name)

    ml_product_forecasts = forecast_ml_products(
        panel, best_ml_model, best_ml_name, selected_features, FORECAST_HORIZON)

    lstm_product_forecasts = pd .DataFrame()
    if "all" in models_to_train or "Bi-LSTM" in models_to_train:
        lstm_product_forecasts, _ = forecast_bilstm_products(
            panel, selected_features, FORECAST_HORIZON)

    combined_product = pd .concat(
        [ml_product_forecasts, lstm_product_forecasts], ignore_index=True)

    last_date = total[MONTH_START_COL].max()
    ml_fmt = format_forecast_output(
        ml_product_forecasts, last_date, FORECAST_HORIZON, is_total=False)
    lstm_fmt = format_forecast_output(
        lstm_product_forecasts, last_date, FORECAST_HORIZON, is_total=False)
    all_products_fmt = format_forecast_output(
        combined_product, last_date, FORECAST_HORIZON, is_total=False)

    total_forecast_sheet = build_total_forecast_sheet(
        total, None, None, ml_product_forecasts, lstm_product_forecasts, FORECAST_HORIZON)

    print("\nExporting results...")
    with pd .ExcelWriter(POWERBI_OUTPUT, engine="openpyxl", mode="w")as writer:
        df .to_excel(writer, index=False, sheet_name="PowerBI_Data")

    with pd .ExcelWriter(FORECAST_OUTPUT, engine="openpyxl", mode="w")as writer:
        total_forecast_sheet .to_excel(
            writer, index=False, sheet_name="Total_Forecast")
        if not ml_fmt .empty:
            ml_fmt .to_excel(writer, index=False,
                             sheet_name="ML_Product_Forecast")
        if not lstm_fmt .empty:
            lstm_fmt .to_excel(writer, index=False,
                               sheet_name="BiLSTM_Product_Forecast")
        if not all_products_fmt .empty:
            all_products_fmt .to_excel(
                writer, index=False, sheet_name="All_Products_Forecast")
        if not ml_eval .empty:
            ml_eval .to_excel(writer, index=False,
                              sheet_name="Model_Evaluation")
        importance_df .to_excel(writer, index=False,
                                sheet_name="Feature_Importance")

    print(f"\nForecast results saved to {FORECAST_OUTPUT}")
    print("Workflow finished successfully.")


if __name__ == "__main__":
    main(models_to_train=["XGB"])
