import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import math
import os
from skopt.plots import plot_objective
from prophet.plot import add_changepoints_to_plot

# Set some matplotlib parameters
mpl.rcParams['figure.figsize'] = (20, 15)
cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))


def visualize_silhouette_plot(k_range, silhouette_scores, optimal_k, save_fig=False):
    '''
    Plot average silhouette score for all samples at different values of k. Use this to determine optimal number of
    clusters (k). The optimal k is the one that maximizes the average Silhouette Score over the range of k provided.
    :param k_range: Range of k explored
    :param silhouette_scores: Average Silhouette Score corresponding to values in k range
    :param optimal_k: The value of k that has the highest average Silhouette Score
    :param save_fig: Flag indicating whether to save the figure
    '''

    # Plot the average Silhouette Score vs. k
    axes = plt.subplot()
    axes.plot(k_range, silhouette_scores)

    # Set plot axis labels, title, and subtitle.
    axes.set_xlabel("k (# of clusters)", labelpad=10, size=15)
    axes.set_ylabel("Average Silhouette Score", labelpad=10, size=15)
    axes.set_xticks(k_range, minor=False)
    axes.axvline(x=optimal_k, linestyle='--')
    axes.set_title("Silhouette Plot", fontsize=25)
    axes.text(0.5, 0.92, "Average Silhouette Score over a range of k-values", size=15, ha='center')

    # Save the image
    if save_fig:
        file_path = cfg['PATHS']['DATA_VISUALIZATIONS'] + 'silhouette_plot_' + \
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png'
        plt.savefig(file_path)
    return


def plot_model_evaluation(forecast_df, model_name, metrics, save_dir=None, figsize=(20,13), save_fig=False, train_date=''):
    '''
    Plot model's predictions on training and test sets, along with key performance metrics.
    :param forecast_df: DataFrame consisting of predicted and ground truth consumption values
    :param model_name: model identifier
    :param metrics: key performance metrics
    :param figsize: size of matplotlib figure
    :param train_date: string representing date model was trained
    '''

    fig = plt.figure(figsize=figsize)
    fig.suptitle(model_name + ' Forecast', fontsize=20)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Plot training performance
    forecast_df[pd.notnull(forecast_df["model"])][["gt", "model"]].plot(color=["black", "green"], title="Training Set Predictions",
                                                                        grid=True, ax=ax1)
    ax1.set(xlabel=None)

    # Plot test performance
    if "test_pred" in forecast_df.columns:
        forecast_df[pd.isnull(forecast_df["model"])][["gt", "forecast", "test_pred"]].plot(color=["black", "red", "yellow"],
                                                                              title="Test Set Forecast", grid=True, ax=ax2)
    else:
        forecast_df[pd.isnull(forecast_df["model"])][["gt", "forecast"]].plot(color=["black", "red"],
                                                                              title="Test Set Forecast", grid=True, ax=ax2)
    ax2.fill_between(x=forecast_df.index, y1=forecast_df['pred_int_low'], y2=forecast_df['pred_int_up'], color='b', alpha=0.2)
    ax2.fill_between(x=forecast_df.index, y1=forecast_df['conf_int_low'], y2=forecast_df['conf_int_up'], color='b', alpha=0.3)
    ax2.set(xlabel=None)

    # Plot residuals
    forecast_df[["residuals", "error"]].plot(ax=ax3, color=["green", "red"], title="Residuals", grid=True)
    ax3.set(xlabel=None)

    # Plot residuals distribution
    forecast_df[["residuals", "error"]].plot(ax=ax4, color=["green", "red"], kind='kde',
                                     title="Residuals Distribution", grid=True)
    ax4.set(ylabel=None)
    print("Training --> Residuals mean:", np.round(metrics['residuals_mean']), " | std:", np.round(metrics['residuals_std']))
    print("Test --> Error mean:", np.round(metrics['error_mean']), " | std:", np.round(metrics['error_std']),
          " | mae:", np.round(metrics['MAE']), " | mape:", np.round(metrics['MAPE'] * 100), "%  | mse:", np.round(metrics['MSE']),
          " | rmse:", np.round(metrics['RMSE']))

    if save_fig:
        save_dir = cfg['PATHS']['FORECAST_VISUALIZATIONS'] if save_dir is None else save_dir
        plt.savefig(save_dir + '/' + model_name + '_eval_' +
                    train_date + '.png')
    return


def correlation_matrix(dataset, save_fig=False):
    '''
    Produces a correlation matrix for a dataset
    :param dataset: A DataFrame
    :save_fig: Flag indicating whether to save the figure
    '''
    corr_mat = dataset.corr()
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))      # Generate mask for upper right triangle
    cmap = sns.diverging_palette(230, 20, as_cmap=True)     # Custom diverging colour map
    fig, axes = plt.subplots()
    sns.heatmap(corr_mat, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})    # Draw a heatmap with mask and correct aspect ratio
    axes.set_title('Correlation Matrix', fontsize=20)
    plt.tight_layout(pad=1.2)
    if save_fig:
        plt.savefig(cfg['PATHS']['DATA_VISUALIZATIONS'] + 'correlation_matrix' +
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return


def client_box_plot(client_df, save_fig=False):
    '''
    Produces a box plot for all features in the dataset
    :param client_df: A DataFrame indexed by client identifier
    :param save_fig: Flag indicating whether to save the figure
    '''

    cat_feats = [f for f in cfg['DATA']['CATEGORICAL_FEATS'] if f in client_df.columns]
    bool_feats = [f for f in cfg['DATA']['BOOLEAN_FEATS'] if f in client_df.columns]
    feats = cat_feats + bool_feats

    n_rows = math.floor(math.sqrt(len(feats)))
    n_cols = math.ceil(math.sqrt(len(feats)))
    fig, axes = plt.subplots(n_rows, n_cols)

    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            sns.boxplot(x=client_df[feats[idx]], y=client_df['CONS_0m_AGO'], palette="Set2", ax=axes[i, j])
            axes[i, j].set_yscale('log')
            axes[i, j].set_title(feats[idx], fontsize=14)
            axes[i, j].set_xticklabels(axes[i, j].get_xticklabels(), rotation=45, ha='right')
            if idx < len(feats) - 1:
                idx += 1
            else:
                break
    fig.suptitle('Box Plots for consumption in recent month grouped by categorical variables', fontsize=20, y=0.99)
    fig.tight_layout(pad=1, rect=(0,0,1,0.95))
    if save_fig:
        plt.savefig(cfg['PATHS']['DATA_VISUALIZATIONS'] + 'client_box_plot' +
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return


def client_cmptn_by_rc_violin_plot(client_df, save_fig=False):
    '''
    Produces a violin plot for consumption by client in the most recent month stratified by rate class
    :param client_df: A DataFrame indexed by client identifier
    :param save_fig: Flag indicating whether to save the figure
    '''
    fig, axes = plt.subplots()
    sns.violinplot(x=client_df['CONS_0m_AGO'], y=client_df['RATE_CLASS'], palette="Set2", scale='area', orient='h',
                   linewidth=0.2, ax=axes)
    axes.set_yticklabels(axes.get_yticklabels(), fontsize=12)
    axes.set_xlabel('Consumption in last month [m^3]', fontsize=20, labelpad=10)
    axes.set_ylabel('Rate Class', fontsize=20, labelpad=10)
    fig.suptitle('Violin plot for consumption in recent month grouped by rate class', fontsize=30)
    fig.tight_layout(pad=1, rect=(0,0.05,1,0.95))
    if save_fig:
        plt.savefig(cfg['PATHS']['DATA_VISUALIZATIONS'] + 'violin_plot_rate_class' +
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return


def visualize_client_dataset_stats(client_df, save_fig=False):
    '''
    Obtain general statistics for features in the client dataset and create a summary figure
    :param client_df: A DataFrame indexed by client identifier
    :param save_fig: Flag indicating whether to save the figure
    '''
    cat_feats = [f for f in cfg['DATA']['CATEGORICAL_FEATS'] if f in client_df.columns]
    bool_feats = [f for f in cfg['DATA']['BOOLEAN_FEATS'] if f in client_df.columns]
    num_feats = [f for f in cfg['DATA']['NUMERICAL_FEATS'] if f in client_df.columns]
    feats = cat_feats + bool_feats + num_feats
    n_feats = len(feats)

    n_rows = math.floor(math.sqrt(n_feats))
    n_cols = math.ceil(math.sqrt(n_feats))
    fig, axes = plt.subplots(n_rows, n_cols)

    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if feats[idx] in num_feats:
                sns.kdeplot(data=client_df, x=feats[idx], palette="Set2", ax=axes[i, j])
                mean = client_df[feats[idx]].mean()
                median = client_df[feats[idx]].median()
                std = client_df[feats[idx]].std()
                axes[i, j].axvline(mean, color='r', linestyle='-', linewidth=0.8, label='mean=' + '{:.1e}'.format(mean))
                axes[i, j].axvline(median, color='g', linestyle='-', linewidth=0.8, label='median=' + '{:.1e}'.format(median))
                axes[i, j].axvline(mean - std, color='r', linestyle='--', linewidth=0.8, label='+/- std' + '{:.1e}'.format(std))
                axes[i, j].axvline(mean + std, color='r', linestyle='--', linewidth=0.8)
                axes[i, j].legend(fontsize=8)
                axes[i, j].set_title(feats[idx], fontsize=14)
            else:
                mode = client_df[feats[idx]].mode()
                sns.countplot(data=client_df, x=feats[idx], ax=axes[i, j], palette='Set3')
                axes[i, j].set_xticklabels(axes[i, j].get_xticklabels(), rotation=45, ha='right')
                axes[i, j].text(0.6, 0.9, 'mode=' + str(mode[0]), transform=axes[i, j].transAxes, fontsize=8)
                axes[i, j].set_title(feats[idx], fontsize=14)
            if idx < n_feats - 1:
                idx += 1
            else:
                break
    fig.suptitle('General statistics for client data', fontsize=20, y=0.99)
    fig.tight_layout(pad=2, rect=(0, 0, 1, 0.95))
    if save_fig:
        plt.savefig(cfg['PATHS']['DATA_VISUALIZATIONS'] + 'client_general_visualization' +
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return


def produce_data_visualizations(preprocessed_path=None, client_path=None):
    '''
    Produces a series of data visualizations for client data and preprocessed consumption data.
    :param preprocessed_path: Path of preprocessed data CSV
    :param client_path: Path of client data CSV
    '''
    if preprocessed_path is None:
        preprocessed_df = pd.read_csv(cfg['PATHS']['PREPROCESSED_DATA'])
    if client_path is None:
        client_df = pd.read_csv(cfg['PATHS']['CLIENT_DATA'])
    plt.clf()
    correlation_matrix(preprocessed_df, save_fig=True)
    plt.clf()
    client_box_plot(client_df, save_fig=True)
    plt.clf()
    visualize_client_dataset_stats(client_df, save_fig=True)
    return


def plot_bayesian_hparam_opt(model_name, hparam_names, search_results, save_fig=False):
    '''
    Plot all 2D hyperparameter comparisons from the logs of a Bayesian hyperparameter optimization.
    :param model_name: Name of the model
    :param hparam_names: List of hyperparameter identifiers
    :param search_results: The object resulting from a Bayesian hyperparameter optimization with the skopt package
    :param save_fig:
    :return:
    '''

    # Abbreviate hyperparameters to improve plot readability
    axis_labels = hparam_names.copy()
    for i in range(len(axis_labels)):
        if len(axis_labels[i]) >= 12:
            axis_labels[i] = axis_labels[i][:4] + '...' + axis_labels[i][-4:]

    # Plot
    axes = plot_objective(result=search_results, dimensions=axis_labels)

    # Create a title
    fig = plt.gcf()
    fig.suptitle('Bayesian Hyperparameter\n Optimization for ' + model_name, fontsize=15, x=0.65, y=0.97)

    # Indicate which hyperparameter abbreviations correspond with which hyperparameter
    hparam_abbrs_text = ''
    for i in range(len(hparam_names)):
        hparam_abbrs_text += axis_labels[i] + ':\n'
    fig.text(0.50, 0.8, hparam_abbrs_text, fontsize=10, style='italic', color='mediumblue')
    hparam_names_text = ''
    for i in range(len(hparam_names)):
        hparam_names_text += hparam_names[i] + '\n'
    fig.text(0.65, 0.8, hparam_names_text, fontsize=10, color='darkblue')

    fig.tight_layout()
    if save_fig:
        plt.savefig(cfg['PATHS']['EXPERIMENT_VISUALIZATIONS'] + 'Bayesian_opt_' + model_name + '_' +
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')


def plot_prophet_components(prophet_model, forecast, save_dir=None, train_date=''):
    '''
    Plot Prophet model's forecast components. This plot visualizes trend, yearly seasonality, weekly seasonality,
    holiday effects
    :param prophet_model: Fitted Prophet model
    :param forecast: A forecast from a Prophet model
    :param train_date: string representing date model was trained
    '''

    fig = prophet_model.plot_components(forecast)
    fig.suptitle('Prophet Model Components', fontsize=15)
    fig.tight_layout(pad=2, rect=(0, 0, 1, 0.95))
    save_dir = cfg['PATHS']['INTERPRETABILITY_VISUALIZATIONS'] if save_dir is None else save_dir
    plt.savefig(save_dir + 'Prophet_components' +
                train_date + '.png')
    return


def plot_prophet_forecast(prophet_model, prophet_pred, save_dir=None, train_date=''):
    '''
    Plot Prophet model's forecast using the Prophet API, including changepoints
    :param prophet_model: Fitted Prophet model
    :param prophet_pred: A forecast from a Prophet model (result of a prophet.predict() call)
    '''

    fig = prophet_model.plot(prophet_pred)
    ax = fig.gca()
    add_changepoints_to_plot(ax, prophet_model, prophet_pred)
    ax = fig.gca()
    ax.set_xlabel('Date')
    ax.set_ylabel('Consumption [m^3]')
    fig.suptitle('Prophet Model Forecast', fontsize=15)
    fig.tight_layout(pad=2, rect=(0, 0, 1, 0.95))
    save_dir = cfg['PATHS']['FORECAST_VISUALIZATIONS'] if save_dir is None else save_dir
    plt.savefig(save_dir + 'Prophet_API_forecast' +
                train_date + '.png')
    return