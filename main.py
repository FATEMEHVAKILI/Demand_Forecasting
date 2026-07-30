import warnings
import numpy as np
import pandas as pd
import jdatetime
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
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np .sqrt(mse),
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


def train_selected_models(X_train, y_train, X_test, y_test,
                          total_train, total_test,
                          models_to_train=["all"], horizon=6):

    print(f"\nTraining selected models: {models_to_train}")
    models = {}
    metrics_list = []
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
            best_rf = rf_search .best_estimator_
            print(f"RF best params: {rf_search.best_params_}")
        else:
            best_rf = rf .fit(X_train, y_train)
        models["Random Forest"] = best_rf
        y_pred = best_rf .predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        metrics["Model"] = "Random Forest"
        metrics_list .append(metrics)

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
                best_xgb = xgb_search .best_estimator_
                print(f"XGB best params: {xgb_search.best_params_}")
            else:
                best_xgb = xgb .set_params(
                    n_estimators=100, max_depth=5, learning_rate=0.05, min_child_weight=1)
                best_xgb .fit(X_train, y_train)
            models["XGBoost"] = best_xgb
            y_pred = best_xgb .predict(X_test)
            metrics = calculate_metrics(y_test, y_pred)
            metrics["Model"] = "XGBoost"
            metrics_list .append(metrics)
        else:
            print("Warning: XGBoost not available.")

    if "all" in models_to_train or "Prophet" in models_to_train:
        if PROPHET_AVAILABLE and len(total_train) >= 6:
            val_size = min(2, len(total_train)//2)
            train_df = total_train .iloc[:-val_size].rename(
                columns={MONTH_START_COL: "ds", "y": "y"})
            val_df = total_train .iloc[-val_size:].rename(
                columns={MONTH_START_COL: "ds", "y": "y"})

            best_mae = float('inf')
            best_prophet = None
            best_params = {}
            for cps in [0.01, 0.05, 0.1]:
                for mode in ['additive', 'multiplicative']:
                    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                                daily_seasonality=False, changepoint_prior_scale=cps,
                                seasonality_mode=mode)
                    m .fit(train_df)
                    future = m .make_future_dataframe(
                        periods=len(val_df), freq="MS")
                    pred = m .predict(future).tail(len(val_df))
                    mae = mean_absolute_error(
                        val_df['y'].values, pred['yhat'].values)
                    if mae < best_mae:
                        best_mae = mae
                        best_prophet = m
                        best_params = {'cps': cps, 'mode': mode}

            final_train = total_train .rename(
                columns={MONTH_START_COL: "ds", "y": "y"})
            final_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                                    daily_seasonality=False,
                                    changepoint_prior_scale=best_params .get(
                                        'cps', 0.05),
                                    seasonality_mode=best_params .get('mode', 'additive'))
            final_prophet .fit(final_train)
            models["Prophet"] = final_prophet

            test_future = final_prophet .make_future_dataframe(
                periods=len(total_test), freq="MS")
            test_pred = final_prophet .predict(
                test_future).tail(len(total_test))
            y_true = total_test['y'].values
            y_pred = test_pred['yhat'].values
            metrics = calculate_metrics(y_true, y_pred)
            metrics["Model"] = "Prophet"
            metrics_list .append(metrics)
        else:
            print("Warning: Prophet not available or not enough data.")

    if "all" in models_to_train or "SARIMA" in models_to_train:
        if SARIMA_AVAILABLE and len(total_train) >= 6:
            val_size = min(2, len(total_train)//2)
            train_series = total_train .iloc[:-val_size]['y'].astype(float)
            val_series = total_train .iloc[-val_size:]['y'].astype(float)

            best_mae = float('inf')
            best_sarima = None
            best_order = None
            orders = [(1, 0, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)]
            seasonal_orders = [(0, 0, 0, 12), (1, 1, 1, 12), (0, 1, 1, 12)]
            for order in orders:
                for s_order in seasonal_orders:
                    try:
                        m = SARIMAX(train_series, order=order, seasonal_order=s_order,
                                    enforce_stationarity=False, enforce_invertibility=False)
                        fitted = m .fit(disp=0, maxiter=50)
                        pred = fitted .forecast(steps=len(val_series))
                        mae = mean_absolute_error(
                            val_series .values, pred .values)
                        if mae < best_mae:
                            best_mae = mae
                            best_sarima = fitted
                            best_order = (order, s_order)
                    except Exception:
                        continue

            full_series = total_train['y'].astype(float)
            final_sarima = SARIMAX(full_series, order=best_order[0], seasonal_order=best_order[1],
                                   enforce_stationarity=False, enforce_invertibility=False)
            final_model = final_sarima .fit(disp=0)
            models["SARIMA"] = final_model

            pred_test = final_model .forecast(steps=len(total_test))
            y_true = total_test['y'].values
            y_pred = pred_test .values
            metrics = calculate_metrics(y_true, y_pred)
            metrics["Model"] = "SARIMA"
            metrics_list .append(metrics)
        else:
            print("Warning: SARIMA not available or not enough data.")

    metrics_df = pd .DataFrame(metrics_list)
    return models, metrics_df


def train_bilstm(X_train, y_train, X_test, y_test, input_size, epochs=20):

    if not TORCH_AVAILABLE:
        return None, None, None, None, None, None

    device = torch .device("cuda"if torch .cuda .is_available()else "cpu")

    param_grid = [
        {'hidden_size': 32, 'num_layers': 1, 'dropout': 0.1,
         'learning_rate': 0.01, 'batch_size': 32},
        {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2,
         'learning_rate': 0.005, 'batch_size': 32},
        {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3,
         'learning_rate': 0.001, 'batch_size': 64}
    ]

    best_model = None
    best_scaler = None
    best_y_scaler = None
    best_mae = float('inf')
    best_test_preds = None
    best_test_actuals = None

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
            pred_scaled = model(X_test_tensor).cpu().numpy()
            pred = y_scaler .inverse_transform(pred_scaled)
            actual = y_scaler .inverse_transform(y_test_scaled)
        current_mae = mean_absolute_error(actual .flatten(), pred .flatten())

        if current_mae < best_mae:
            best_mae = current_mae
            best_model = model
            best_scaler = scaler
            best_y_scaler = y_scaler
            best_test_preds = pred .flatten()
            best_test_actuals = actual .flatten()

    metrics = calculate_metrics(best_test_actuals, best_test_preds)
    return best_model, best_scaler, best_y_scaler, metrics, best_test_actuals, best_test_preds


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
        product_name = group["نام کالا"].iloc[-1]

        last_row = group .iloc[-1][selected_features].to_dict()

        for horizon_step, month_start in enumerate(future_months, start=1):

            if "ماه میلادی" in selected_features:
                last_row["ماه میلادی"] = month_start .month
            if "فصل شمسی" in selected_features:
                last_row["فصل شمسی"] = get_persian_features(month_start)[
                    "فصل شمسی"]
            if "سال شمسی" in selected_features:
                persian_year = jdatetime .date .fromgregorian(
                    year=month_start .year, month=month_start .month, day=month_start .day).year
                last_row["سال شمسی"] = persian_year

            last_values = targets[-3:]if len(targets) >= 3 else targets
            if "lag_1" in selected_features:
                last_row["lag_1"] = targets[-1]if len(targets) >= 1 else 0.0
            if "lag_2" in selected_features:
                last_row["lag_2"] = targets[-2]if len(targets) >= 2 else 0.0
            if "lag_3" in selected_features:
                last_row["lag_3"] = targets[-3]if len(targets) >= 3 else 0.0
            if "rolling_mean_3" in selected_features:
                last_row["rolling_mean_3"] = float(
                    np .mean(last_values))if last_values else 0.0
            if "rolling_std_3" in selected_features:
                last_row["rolling_std_3"] = float(
                    np .std(last_values, ddof=0))if len(last_values) > 1 else 0.0

            feature_df = pd .DataFrame([last_row])[selected_features].fillna(0)
            predicted = max(0.0, float(model .predict(feature_df)[0]))
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
        return pd .DataFrame(), pd .DataFrame(), None

    records = []
    per_product_metrics = []
    all_test_actuals = []
    all_test_preds = []

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

        model, scaler, y_scaler, test_metrics, test_actuals, test_preds = train_bilstm(
            X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:],
            input_size=X .shape[1], epochs=10
        )
        if model is None:
            continue

        if test_metrics:
            per_product_metrics .append(
                {"Product": product_code, **test_metrics})
            all_test_actuals .extend(test_actuals .tolist())
            all_test_preds .extend(test_preds .tolist())

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

    overall_metrics = None
    if all_test_actuals and all_test_preds:
        overall_metrics = calculate_metrics(
            np .array(all_test_actuals), np .array(all_test_preds))

    return pd .DataFrame(records), pd .DataFrame(per_product_metrics), overall_metrics


def build_total_forecast_sheet(total, model_forecasts_dict):

    frames = []

    hist = total .rename(columns={MONTH_START_COL: "ds", "y": "forecast"})
    hist["Persian Month"] = hist["ds"].apply(
        lambda x: f"{jdatetime.date.fromgregorian(date=x).year}/{jdatetime.date.fromgregorian(date=x).month:02d}"if pd .notnull(
            x)else ""
    )
    hist["Persian Year"] = hist["ds"].apply(
        lambda x: jdatetime .date .fromgregorian(
            date=x).year if pd .notnull(x)else 0
    )
    hist = hist[hist["Persian Year"] >= 1404].copy()
    hist = hist .drop(columns=["Persian Year"])

    hist["Model"] = "Actual"
    frames .append(hist[["Persian Month", "Model", "forecast"]])

    for model_name, df in model_forecasts_dict .items():
        if df is None or df .empty:
            continue
        df_out = df .copy()
        df_out["Model"] = model_name
        df_out["Persian Month"] = df_out["ds"].apply(
            lambda x: f"{jdatetime.date.fromgregorian(date=x).year}/{jdatetime.date.fromgregorian(date=x).month:02d}"if pd .notnull(
                x)else ""
        )
        frames .append(df_out[["Persian Month", "Model", "forecast"]])

    combined = pd .concat(frames, ignore_index=True)
    combined["_sort"] = combined["Persian Month"].apply(
        lambda x: tuple(map(int, x .split("/")))if x else (0, 0)
    )
    combined = combined .sort_values(["Model", "_sort"]).drop(
        columns=["_sort"]).reset_index(drop=True)
    combined = combined .rename(
        columns={"forecast": "Forecasting Value", "Model": "Model Name"})
    return combined


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

    with pd .ExcelWriter(POWERBI_OUTPUT, engine="openpyxl", mode="w")as writer:
        df .to_excel(writer, index=False, sheet_name="PowerBI_Data")

        abc_cols = [col for col in df .columns if col in ["ماه شمسی", PRODUCT_CODE_COL, "کلاس ABC", "رتبه کالا در ماه",
                                                          "مجموع مصرف کالا در ماه", "تعداد تراکنش کالا در ماه",
                                                          "سهم درصد کالا از مصرف ماه", "درصد تجمعی ماه"]]
        if abc_cols:
            df[abc_cols].drop_duplicates().to_excel(
                writer, index=False, sheet_name="ABC_Analysis")
        else:
            print("Warning: ABC columns not found for export.")
    print(f"PowerBI data and ABC analysis combined into {POWERBI_OUTPUT}")

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

    importance_df, selected_features = analyze_feature_importance(
        panel, feature_candidates, target_col="target"
    )

    print(f"\nProceeding with {len(selected_features)} selected features.")
    X_train = train_panel[selected_features].fillna(0)
    y_train = train_panel["target"]
    X_test = test_panel[selected_features].fillna(0)
    y_test = test_panel["target"]

    total = panel .groupby(MONTH_START_COL, as_index=False)[
        "target"].sum().sort_values(MONTH_START_COL).reset_index(drop=True)
    total = total .rename(columns={"target": "y"})
    total_train = total[total[MONTH_START_COL].isin(train_months)]
    total_test = total[total[MONTH_START_COL].isin(test_months)]

    models, ml_eval = train_selected_models(
        X_train, y_train, X_test, y_test,
        total_train, total_test,
        models_to_train=models_to_train, horizon=FORECAST_HORIZON
    )

    model_forecasts = {}
    future_dates = pd .date_range(total[MONTH_START_COL].max()+pd .DateOffset(months=1),
                                  periods=FORECAST_HORIZON, freq="MS")

    for model_name, model in models .items():
        if model_name in ["Random Forest", "XGBoost"]:
            prod_forecast = forecast_ml_products(
                panel, model, model_name, selected_features, FORECAST_HORIZON)
            if not prod_forecast .empty:
                total_forecast = prod_forecast .groupby(
                    "ds")["forecast"].sum().reset_index()
                model_forecasts[model_name] = total_forecast
        elif model_name == "Prophet" and PROPHET_AVAILABLE:
            future = model .make_future_dataframe(
                periods=FORECAST_HORIZON, freq="MS")
            pred = model .predict(future).tail(FORECAST_HORIZON)
            df_forecast = pd .DataFrame(
                {"ds": pred["ds"], "forecast": pred["yhat"]})
            model_forecasts["Prophet"] = df_forecast
        elif model_name == "SARIMA" and SARIMA_AVAILABLE:
            pred = model .forecast(steps=FORECAST_HORIZON)
            df_forecast = pd .DataFrame(
                {"ds": future_dates, "forecast": pred .values})
            model_forecasts["SARIMA"] = df_forecast

    if "all" in models_to_train or "Bi-LSTM" in models_to_train:
        lstm_prod, _, _ = forecast_bilstm_products(
            panel, selected_features, FORECAST_HORIZON)
        if not lstm_prod .empty:
            lstm_total = lstm_prod .groupby(
                "ds")["forecast"].sum().reset_index()
            model_forecasts["Bi-LSTM"] = lstm_total

    total_forecast_sheet = build_total_forecast_sheet(total, model_forecasts)

    all_product_frames = []
    for model_name, model in models .items():
        if model_name in ["Random Forest", "XGBoost"]:
            prod_fc = forecast_ml_products(
                panel, model, model_name, selected_features, FORECAST_HORIZON)
            if not prod_fc .empty:
                all_product_frames .append(prod_fc)
    if "all" in models_to_train or "Bi-LSTM" in models_to_train:
        lstm_prod, _, _ = forecast_bilstm_products(
            panel, selected_features, FORECAST_HORIZON)
        if not lstm_prod .empty:
            all_product_frames .append(lstm_prod)

    combined_product = pd .concat(
        all_product_frames, ignore_index=True)if all_product_frames else pd .DataFrame()

    def format_product(df):
        if df .empty:
            return df
        df_out = df .copy()
        df_out = df_out .rename(
            columns={"forecast": "Forecasting Value", "Model": "Model Name"})

        cols = [PRODUCT_CODE_COL, "نام کالا", "ماه شمسی",
                "Model Name", "Forecasting Value"]

        if "horizon" in df_out .columns:
            df_out = df_out .drop(columns=["horizon"])
        return df_out[cols]

    ml_fmt = format_product(combined_product[combined_product["Model"].isin(
        ["Random Forest", "XGBoost"])])if not combined_product .empty else pd .DataFrame()
    lstm_fmt = format_product(
        combined_product[combined_product["Model"] == "Bi-LSTM"])if not combined_product .empty else pd .DataFrame()
    all_products_fmt = format_product(
        combined_product)if not combined_product .empty else pd .DataFrame()

    all_metrics = ml_eval .copy()
    if "all" in models_to_train or "Bi-LSTM" in models_to_train:

        _, _, overall_lstm = forecast_bilstm_products(
            panel, selected_features, FORECAST_HORIZON)
        if overall_lstm is not None:
            lstm_row = {"Model": "Bi-LSTM"}
            lstm_row .update(overall_lstm)
            all_metrics = pd .concat(
                [all_metrics, pd .DataFrame([lstm_row])], ignore_index=True)

    print("\nExporting results...")
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
        if not all_metrics .empty:
            all_metrics .to_excel(writer, index=False,
                                  sheet_name="Model_Evaluation")
        importance_df .to_excel(writer, index=False,
                                sheet_name="Feature_Importance")

    print(f"\nForecast results saved to {FORECAST_OUTPUT}")
    print("Workflow finished successfully.")


if __name__ == "__main__":
    main(models_to_train=["XGB"])
