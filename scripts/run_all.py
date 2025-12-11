
import os
from src.data_loader import load_dataset
from src.preprocess import prepare_dataset_for_recall, prepare_userinfo
from src.evaluate import precision_recall_per_user, compute_map_ndcg, plot_pr
from src.export_excel import (
    write_pr_excel, write_map_excel,
    write_satisfaction_excel, write_sentiment_excel, write_revenue_excel
)
from src.sentiment import sentiment_accuracy
import pandas as pd
import numpy as np

# Output folder
OUT = os.path.join(os.getcwd(), 'outputs')
os.makedirs(OUT, exist_ok=True)

def main():
    print('Loading Djanet tourism dataset...')
    df = load_dataset()  # Loads Based_Djanet_Dataset.xlsx

    # -------------------------------------------------------
    # Prepare the dataset for different evaluation tasks
    # -------------------------------------------------------
    recall_df = prepare_dataset_for_recall(df)
    user_info = prepare_userinfo(df)

    # =======================================================
    # 1. Precision vs Recall
    # =======================================================
    print("Computing Precision/Recall...")
    per_user_pr, pr_summary = precision_recall_per_user(recall_df)

    pr_path = os.path.join(OUT, 'Precision_vs_recall.xlsx')
    write_pr_excel(per_user_pr, pr_summary, pr_path)

    plot_pr(pr_summary, os.path.join(OUT, 'Precision_vs_recall_with_chart.png'))

    print("Precision/Recall exported ->", pr_path)

    # =======================================================
    # 2. MAP and nDCG (AP_vs_nDCG.xlsx + MAP_vs_DCG_Distribution.xlsx)
    # =======================================================
    print("Computing MAP and nDCG...")
    per_user_map, map_summary = compute_map_ndcg(recall_df)

    map_path = os.path.join(OUT, 'AP_vs_nDCG.xlsx')
    write_map_excel(per_user_map, map_summary, map_path)

    map_dist_path = os.path.join(OUT, 'MAP_vs_DCG_Distribution.xlsx')
    write_map_excel(per_user_map, map_summary, map_dist_path)

    print("MAP/nDCG exported ->", map_path)

    # =======================================================
    # 3. Satisfaction vs SUS & Mean SUS vs NPS
    # =======================================================
    print("Computing Satisfaction vs SUS...")
    ux = user_info[['User_ID', 'Satisfaction_0_1', 'SUS', 'NPS']].dropna()\
            .rename(columns={'Satisfaction_0_1': 'Satisfaction'})

    mean_sus_by_nps = ux.groupby('NPS').SUS.mean().reset_index()\
                        .rename(columns={'SUS': 'mean_SUS'})

    sat_path = os.path.join(OUT, 'Satisfaction_vs_SUS.xlsx')
    write_satisfaction_excel(ux, mean_sus_by_nps, sat_path)

    mean_sus_nps_path = os.path.join(OUT, 'Mean_SUS_vs_NPS.xlsx')
    write_satisfaction_excel(ux, mean_sus_by_nps, mean_sus_nps_path)

    print("Satisfaction exported ->", sat_path)

    # =======================================================
    # 4. Sentiment accuracy vs resolution time
    # =======================================================
    print("Computing Sentiment Accuracy...")
    overall_sent_acc, by_source = sentiment_accuracy(user_info)

    sent_df = by_source.copy()
    sent_df['mean_resolution_time'] = (
        user_info.groupby('Booking_Source').Resolution_Time_Min.mean().values
    )

    sent_path = os.path.join(OUT, 'Sentiment_vs_Resolution.xlsx')
    write_sentiment_excel(sent_df, sent_path)

    print("Sentiment exported ->", sent_path)

    # =======================================================
    # 5. Revenue by booking source
    # =======================================================
    print("Computing Revenue vs Booking Source...")
    rev = user_info.groupby('Booking_Source').agg(
        total_revenue=('Price_USD', 'sum'),
        mean_price=('Price_USD', 'mean'),
        bookings=('Price_USD', 'count')
    ).reset_index()

    rev_path = os.path.join(OUT, 'Revenue_vs_booking_source.xlsx')
    write_revenue_excel(rev, rev_path)

    print("Revenue exported ->", rev_path)

if __name__ == '__main__':
    main()
