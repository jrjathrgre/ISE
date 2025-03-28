import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np


def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 5
    train_frac = 0.7
    random_seed = 1
    tree_results = pd.DataFrame(columns=['system', 'dataset', 'repeat', 'MAPE', 'MAE', 'RMSE'])

    for current_system in systems:
        datasets_location = f'datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(
                f'\n> System: {current_system}, Dataset: {csv_file}, Training data fraction: {train_frac}, Number of repeats: {num_repeats}')

            data = pd.read_csv(os.path.join(datasets_location, csv_file))

            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                label_encoders[col] = le

            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                model = RandomForestRegressor(n_estimators=100, random_state=random_seed)
                model.fit(training_X, training_Y)
                predictions = model.predict(testing_X)

                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

                new_row = pd.DataFrame({
                    'system': [current_system],
                    'dataset': [csv_file],
                    'repeat': [current_repeat],
                    'MAPE': [mape],
                    'MAE': [mae],
                    'RMSE': [rmse]
                })
                tree_results = pd.concat([tree_results, new_row], ignore_index=True)

            print('Average MAPE: {:.2f}'.format(np.mean(metrics['MAPE'])))
            print('Average MAE: {:.2f}'.format(np.mean(metrics['MAE'])))
            print('Average RMSE: {:.2f}'.format(np.mean(metrics['RMSE'])))

    if not os.path.exists('tree_results.csv'):
        tree_results.to_csv('tree_results.csv', index=False)


if __name__ == "__main__":
    main()
