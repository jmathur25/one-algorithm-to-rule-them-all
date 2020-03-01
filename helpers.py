def na_handler(df, drop=0.5):
    nas = df.isna().sum() / len(df)
    
    medians = {}
    dropped = []
    for col in nas.index:
        if nas[col] >= drop:
            df = df.drop(col, axis=1)
            dropped.append(col)
        elif nas[col] > 0:
            medians[col] = df[col].median()
            df[col] = df[col].fillna(medians[col])
    
    return df, medians, dropped