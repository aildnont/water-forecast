from fbprophet import Prophet
import pandas as pd
from src.models.model import ModelStrategy

class ProphetModel(ModelStrategy):

    def __init__(self, hparams):
        univariate = True
        name = 'Prophet'

        # Build DataFrame of local holidays
        holiday_dfs = []
        for holiday in hparams['HOLIDAYS']:
            holiday_dfs.append(pd.DataFrame({
                'holiday': holiday,
                'ds': pd.to_datetime(hparams['HOLIDAYS'][holiday]),
                'lower_window': 0,
                'upper_window': 1}))
        local_holidays = pd.concat(holiday_dfs)

        model = Prophet(yearly_seasonality=True, holidays=local_holidays)
        model.add_country_holidays(country_name=hparams['COUNTRY'])   # Add country-wide holidays
        super(ProphetModel, self).__init__(model, univariate, name)


    def fit(self, dataset):
        '''
        Fits a Prophet forecasting model
        :param dataset: A Pandas DataFrame with 2 columns: Date and Consumption
        '''
        if dataset.shape[1] != 2:
            raise Exception('Univariate models cannot fit with datasets with more than 1 feature.')
        dataset.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
        self.model.fit(dataset)
        return


    def evaluate(self, train_set, test_set, save_dir=None):
        '''
        Evaluates performance of Prophet model on test set
        :param train_set: A Pandas DataFrame with 2 columns: Date and Consumption
        :param test_set: A Pandas DataFrame with 2 columns: Date and Consumption
        '''
        train_set.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
        test_set.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
        df_prophet = self.model.make_future_dataframe(periods=test_set.shape[0], include_history=True, freq='D')
        df_prophet = self.model.predict(df_prophet)
        df_train = train_set.merge(df_prophet[["ds", "yhat"]],
                                    how="left").rename(columns={'yhat': 'model', 'y': 'gt'}).set_index("ds")
        df_test = test_set.merge(df_prophet[["ds", "yhat"]],
                                  how="left").rename(columns={'yhat': 'forecast', 'y': 'gt'}).set_index("ds")
        df_forecast = df_train.append(df_test)
        test_metrics = self.evaluate_forecast(df_forecast, save_dir=save_dir)
        return test_metrics


    def forecast(self, days):
        future_dates = self.model.make_future_dataframe(periods=days)
        return self.model.predict(future_dates)



