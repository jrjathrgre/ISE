import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from torch.utils.data import DataLoader, TensorDataset, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Deep(nn.Module):
    def __init__(self, input_size, hidden_layers=None, l1_lambda=0.01):
        super().__init__()
        self.l1_lambda = l1_lambda

        layers = []
        prev_size = input_size
        if hidden_layers:
            for size in hidden_layers:
                layers.append(nn.Linear(prev_size, size))
                layers.append(nn.ReLU())
                prev_size = size
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, 1)

    def forward(self, x):
        x = self.hidden(x)
        return self.output(x)

    def l1_loss(self):
        if len(self.hidden) > 0:
            return self.l1_lambda * torch.norm(self.hidden[0].weight, p=1)
        return 0


def objective(trial, train_loader, val_loader, input_size):
    params = {
        'n_layers': trial.suggest_int('n_layers', 1, 5),
        'hidden_units': [trial.suggest_categorical(f'units_{i}', [64, 128, 256])
                         for i in range(trial.suggest_int('n_layers', 1, 5))],
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'l1_lambda': trial.suggest_float('l1_lambda', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    }

    model = Deep(
        input_size,
        hidden_layers=params['hidden_units'],
        l1_lambda=params['l1_lambda']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(1000):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y) + model.l1_loss()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    return best_loss


class DaLLearner:
    def __init__(self, n_partitions=3):
        self.n_partitions = n_partitions
        self.partition_model = None
        self.local_models = {}
        self.input_size = None

    def fit(self, x, y):
        self.partition_model = DecisionTreeRegressor(
            max_leaf_nodes=self.n_partitions,
            min_samples_leaf=50
        )
        self.partition_model.fit(x.cpu().numpy(), y.cpu().numpy())
        leaf_ids = self.partition_model.apply(x.cpu().numpy()).astype(int)  # 修改：X.cpu().numpy()

        self.input_size = x.shape[1]
        for leaf_id in np.unique(leaf_ids):
            mask = (leaf_ids == leaf_id)
            x_leaf = x[mask]
            y_leaf = y[mask]

            kf = KFold(n_splits=3, shuffle=True)
            train_idx, val_idx = next(kf.split(x_leaf))

            train_dataset = TensorDataset(x_leaf[train_idx], y_leaf[train_idx])
            val_dataset = TensorDataset(x_leaf[val_idx], y_leaf[val_idx])

            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: objective(
                    trial,
                    DataLoader(train_dataset, batch_size=trial.suggest_categorical('batch_size', [16, 32, 64])),
                    DataLoader(val_dataset, batch_size=32),
                    self.input_size
                ),
                n_trials=10
            )

            best_params = study.best_params
            model = Deep(
                self.input_size,
                hidden_layers=[best_params[f'units_{i}']
                               for i in range(best_params['n_layers'])],
                l1_lambda=best_params['l1_lambda']
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

            train_loader = DataLoader(train_dataset,
                                      batch_size=best_params['batch_size'])
            model = self._train_model(model, train_loader, optimizer)
            self.local_models[leaf_id] = model

    def _train_model(self, model, loader, optimizer, epochs=1000):
        criterion = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y) + model.l1_loss()
                loss.backward()
                optimizer.step()
        return model

    def predict(self, X):
        leaf_ids = self.partition_model.apply(X.cpu().numpy()).astype(int)  # 修改：X.cpu().numpy()

        predictions = []
        for lid in np.unique(leaf_ids):
            mask = (leaf_ids == lid)
            x_part = X[mask]
            model = self.local_models[lid]
            with torch.no_grad():
                preds = model(x_part.to(device)).cpu().numpy()
            predictions.extend(preds.flatten().tolist())

        sorted_indices = np.argsort(np.concatenate([np.where(mask)[0] for lid, mask in zip(np.unique(leaf_ids),
                                                                                           [leaf_ids == lid for lid in
                                                                                            np.unique(leaf_ids)])]))
        return np.array(predictions)[sorted_indices]


class DynamicPreprocessor:
    def __init__(self):
        self.preprocessor = None

    def fit_transform(self, X):
        numeric = []
        categorical = []
        binary = []

        for col in X.columns:
            unique = X[col].unique()
            if len(unique) == 2 and set(unique) <= {0, 1}:
                binary.append(col)
            elif X[col].dtype in [np.int64, np.float64]:
                numeric.append(col)
            else:
                categorical.append(col)

        self.preprocessor = ColumnTransformer([
            ('binary', 'passthrough', binary),
            ('num', StandardScaler(), numeric),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
        ])
        return self.preprocessor.fit_transform(X)

    def transform(self, X):
        return self.preprocessor.transform(X)


def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    results = []

    for system in systems:
        data_dir = f'datasets/{system}'
        for csv_file in os.listdir(data_dir):
            if not csv_file.endswith('.csv'):
                continue

            data = pd.read_csv(os.path.join(data_dir, csv_file))
            x_raw = data.iloc[:, :-1]
            y = data.iloc[:, -1].values.reshape(-1, 1)

            dataset_metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            preprocessor = DynamicPreprocessor()
            x = preprocessor.fit_transform(x_raw)
            x_tensor = torch.FloatTensor(x).to(device)
            y_tensor = torch.FloatTensor(y).to(device)

            kf = KFold(n_splits=5, shuffle=True)
            for fold, (train_idx, test_idx) in enumerate(kf.split(x_tensor)):
                x_train, x_test = x_tensor[train_idx], x_tensor[test_idx]
                y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

                learner = DaLLearner(n_partitions=5)
                learner.fit(x_train, y_train)

                preds = learner.predict(x_test)
                mape = mean_absolute_percentage_error(y_test.cpu(), preds)
                mae = mean_absolute_error(y_test.cpu(), preds)
                rmse = np.sqrt(mean_squared_error(y_test.cpu(), preds))

                dataset_metrics['MAPE'].append(mape)
                dataset_metrics['MAE'].append(mae)
                dataset_metrics['RMSE'].append(rmse)

                results.append({
                    'system': system,
                    'dataset': csv_file,
                    'fold': fold,
                    'MAPE': mape,
                    'MAE': mae,
                    'RMSE': rmse
                })

            print(f'\n> System: {system}, Dataset: {csv_file}')
            print('Average MAPE: {:.4f}'.format(np.mean(dataset_metrics['MAPE'])))
            print('Average MAE: {:.2f}'.format(np.mean(dataset_metrics['MAE'])))
            print('Average RMSE: {:.2f}'.format(np.mean(dataset_metrics['RMSE'])))

    if not os.path.exists('results.csv'):
        df = pd.DataFrame(results)
        df.to_csv('results.csv', index=False)


if __name__ == "__main__":
    main()
