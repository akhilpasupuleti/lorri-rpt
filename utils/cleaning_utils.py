def iqr_filter(df, col, k=1.5):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + k * iqr
    return df[df[col] <= upper_limit]
