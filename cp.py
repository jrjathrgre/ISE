import pandas as pd
import matplotlib.pyplot as plt

df_baseline = pd.read_csv("baseline_results.csv")
df_deepperf = pd.read_csv("results.csv")
df_tree = pd.read_csv("tree_results.csv")


def preprocess(df):
    return df.groupby(['system', 'dataset']).agg({
        'MAPE': 'mean',
        'MAE': 'mean',
        'RMSE': 'mean'
    }).reset_index()


baseline = preprocess(df_baseline).rename(columns={
    'MAPE': 'MAPE_baseline',
    'MAE': 'MAE_baseline',
    'RMSE': 'RMSE_baseline'
})

deepperf = preprocess(df_deepperf).rename(columns={
    'MAPE': 'MAPE_deepperf',
    'MAE': 'MAE_deepperf',
    'RMSE': 'RMSE_deepperf'
})

tree_model = preprocess(df_tree).rename(columns={
    'MAPE': 'MAPE_tree',
    'MAE': 'MAE_tree',
    'RMSE': 'RMSE_tree'
})

merged = pd.merge(baseline, deepperf, on=['system', 'dataset'])
merged = pd.merge(merged, tree_model, on=['system', 'dataset'])


def calc_improvement(new, old):
    return ((old - new) / old) * 100 if old != 0 else 0


weights = {
    'MAPE': 0.4,
    'MAE': 0.3,
    'RMSE': 0.3
}

merged['Improvement_deepperf'] = merged.apply(
    lambda x: (
            weights['MAPE'] * calc_improvement(x['MAPE_deepperf'], x['MAPE_baseline']) +
            weights['MAE'] * calc_improvement(x['MAE_deepperf'], x['MAE_baseline']) +
            weights['RMSE'] * calc_improvement(x['RMSE_deepperf'], x['RMSE_baseline'])
    ), axis=1
)

merged['Improvement_tree'] = merged.apply(
    lambda x: (
            weights['MAPE'] * calc_improvement(x['MAPE_tree'], x['MAPE_baseline']) +
            weights['MAE'] * calc_improvement(x['MAE_tree'], x['MAE_baseline']) +
            weights['RMSE'] * calc_improvement(x['RMSE_tree'], x['RMSE_baseline'])
    ), axis=1
)

fig, ax = plt.subplots(figsize=(18, 10))
ax.axis('off')

columns = [
    'System', 'Dataset',
    'DECART\nMAPE', 'DECART\nMAE', 'DECART\nRMSE',
    'DeepPerf\nMAPE', 'DeepPerf\nMAE', 'DeepPerf\nRMSE', 'DeepPerf\nImp(%)',
    'TreeModel\nMAPE', 'TreeModel\nMAE', 'TreeModel\nRMSE', 'TreeModel\nImp(%)'
]

cell_data = []
for _, row in merged.iterrows():
    cell_data.append([
        row['system'],
        row['dataset'],
        f"{row['MAPE_baseline']:.2f}",
        f"{row['MAE_baseline']:.2f}",
        f"{row['RMSE_baseline']:.2f}",
        f"{row['MAPE_deepperf']:.2f}",
        f"{row['MAE_deepperf']:.2f}",
        f"{row['RMSE_deepperf']:.2f}",
        f"{row['Improvement_deepperf']:.1f}%",
        f"{row['MAPE_tree']:.2f}",
        f"{row['MAE_tree']:.2f}",
        f"{row['RMSE_tree']:.2f}",
        f"{row['Improvement_tree']:.1f}%"
    ])

table = ax.table(
    cellText=cell_data,
    colLabels=columns,
    cellLoc='center',
    loc='center',
    colColours=['#f0f0f0'] * len(columns)
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

col_widths = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
for i, width in enumerate(col_widths):
    table.auto_set_column_width([i])

prev_system = None
start_row = 1

for i in range(len(cell_data)):
    current_system = cell_data[i][0]

    if current_system == prev_system:
        for col in [0]:
            table.get_celld()[(i + 1, col)].set_visible(False)
            table.get_celld()[(i + 1, col)].set_height(0)
    else:
        prev_system = current_system

    imp_deep = float(cell_data[i][8].replace('%', ''))
    imp_tree = float(cell_data[i][12].replace('%', ''))

    table[i + 1, 8].set_facecolor('#d4edda' if imp_deep >= 0 else '#f8d7da')
    table[i + 1, 12].set_facecolor('#d4edda' if imp_tree >= 0 else '#f8d7da')

output_df = pd.DataFrame(cell_data, columns=columns)
output_df.to_csv('model_comparison_results.csv', index=False)

plt.savefig('performance_comparison_models.png', bbox_inches='tight', dpi=300)
plt.show()
