from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from src.models.model import ModelStrategy

class ProphetModel(ModelStrategy):

    def __init__(self):
        univariate = True
        model = Prophet(yearly_seasonality=True)
        model.add_country_holidays(country_name='CA')
        print(model.holidays)
        name = 'Prophet'
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
        self.evaluate_forecast(df_forecast, save_dir=save_dir)
        return


    def forecast(self, periods):
        future_dates = self.model.make_future_dataframe(periods=periods)
        return self.model.predict(future_dates)



