#!/usr/bin/env python3
import os
from src.data_loader import load_recall_ready, load_ultimate
from src.preprocess import prepare_recall_dataset, prepare_ultimate_for_userinfo
from src.evaluate import precision_recall_per_user, compute_map_ndcg, plot_pr
from src.export_excel import write_pr_excel, write_map_excel, write_satisfaction_excel, write_sentiment_excel, write_revenue_excel
from src.sentiment import sentiment_accuracy
import pandas as pd
import numpy as np

OUT = os.path.join(os.getcwd(),'outputs')
os.makedirs(OUT, exist_ok=True)

def main():
    print('Loading datasets...')
    recall = load_recall_ready()
    ultimate = load_ultimate()
    recall = prepare_recall_dataset(recall)
    user_info = prepare_ultimate_for_userinfo(ultimate)

    # Precision/Recall
    per_user_pr, pr_summary = precision_recall_per_user(recall)
    pr_path = os.path.join(OUT, 'precision_vs_recall_FINAL.xlsx')
    write_pr_excel(per_user_pr, pr_summary, pr_path)
    plot_pr(pr_summary, os.path.join(OUT,'precision_vs_recall_FINAL.png'))
    print('PR done ->', pr_path)

    # MAP / nDCG
    per_user_map, map_summary = compute_map_ndcg(recall)
    map_path = os.path.join(OUT, 'map_ndcg_recall_ready_with_chart.xlsx')
    write_map_excel(per_user_map, map_summary, map_path)
    print('MAP/nDCG done ->', map_path)

    # Satisfaction vs SUS and Mean SUS vs NPS
    ux = user_info[['User_ID','Satisfaction_0_1','SUS','NPS']].dropna().rename(columns={'Satisfaction_0_1':'Satisfaction'})
    mean_sus_by_nps = ux.groupby('NPS').SUS.mean().reset_index().rename(columns={'SUS':'mean_SUS'})
    sat_path = os.path.join(OUT, 'satisfaction_vs_sus_recall_ready_with_chart.xlsx')
    write_satisfaction_excel(ux, mean_sus_by_nps, sat_path)
    print('Satisfaction done ->', sat_path)

    # Sentiment accuracy vs resolution time
    overall_sent_acc, by_source = sentiment_accuracy(user_info)
    sent_df = by_source.copy()
    sent_df['mean_resolution_time'] = user_info.groupby('Booking_Source').Resolution_Time_Min.mean().values
    sent_path = os.path.join(OUT, 'sentiment_accuracy_vs_resolution_recall_ready_with_chart.xlsx')
    write_sentiment_excel(sent_df, sent_path)
    print('Sentiment done ->', sent_path)

    # Revenue by booking source
    rev = user_info.groupby('Booking_Source').agg(total_revenue=('Price_USD','sum'), mean_price=('Price_USD','mean'), bookings=('Price_USD','count')).reset_index()
    rev_path = os.path.join(OUT, 'revenue_by_booking_source_recall_ready_with_chart.xlsx')
    write_revenue_excel(rev, rev_path)
    print('Revenue done ->', rev_path)

if __name__ == '__main__':
    main()
