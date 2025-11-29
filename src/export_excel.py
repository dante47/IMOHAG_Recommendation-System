import pandas as pd
import os
from openpyxl import load_workbook
from openpyxl.chart import ScatterChart, Reference, Series, BarChart, LineChart

OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def write_pr_excel(per_user_pr_df, pr_summary_df, path):
    with pd.ExcelWriter(path) as writer:
        per_user_pr_df.to_excel(writer, sheet_name='per_user_PR', index=False)
        pr_summary_df.to_excel(writer, sheet_name='summary_PR', index=False)

def write_map_excel(per_user_map_df, summary, path):
    with pd.ExcelWriter(path) as writer:
        per_user_map_df.to_excel(writer, sheet_name='per_user_metrics', index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name='summary', index=False)

def write_satisfaction_excel(ux_df, mean_sus_by_nps, path):
    with pd.ExcelWriter(path) as writer:
        ux_df.to_excel(writer, sheet_name='per_user_experience', index=False)
        mean_sus_by_nps.to_excel(writer, sheet_name='mean_SUS_by_NPS', index=False)

def write_sentiment_excel(sent_summary_df, path):
    with pd.ExcelWriter(path) as writer:
        sent_summary_df.to_excel(writer, sheet_name='data', index=False)

def write_revenue_excel(rev_df, path):
    with pd.ExcelWriter(path) as writer:
        rev_df.to_excel(writer, sheet_name='data', index=False)

# Charts can be added later in Excel; this module focuses on reliable data export.
if __name__ == '__main__':
    print('Export helpers ready')
