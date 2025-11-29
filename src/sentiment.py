import pandas as pd
import numpy as np

def predict_sentiment_from_satisfaction(s):
    if s >= 0.66:
        return 'positive'
    elif s <= 0.33:
        return 'negative'
    else:
        return 'neutral'

def sentiment_accuracy(user_info_df):
    df = user_info_df.copy()
    df['Pred_Sent'] = df['Satisfaction_0_1'].apply(predict_sentiment_from_satisfaction)
    df['Sentiment_Correct'] = (df['Pred_Sent'].str.lower() == df['Sentiment_Label'].astype(str).str.lower())
    overall = df['Sentiment_Correct'].mean()
    by_source = df.groupby('Booking_Source').Sentiment_Correct.mean().reset_index().rename(columns={'Sentiment_Correct':'sentiment_accuracy'})
    return overall, by_source

if __name__ == '__main__':
    print('Sentiment utilities loaded')
