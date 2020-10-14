import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
import yaml
import os

# Set some matplotlib parameters
mpl.rcParams['figure.figsize'] = (12, 10)
cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))


def visualize_silhouette_plot(k_range, silhouette_scores, optimal_k, file_path=None):
    '''
    Plot average silhouette score for all samples at different values of k. Use this to determine optimal number of
    clusters (k). The optimal k is the one that maximizes the average Silhouette Score over the range of k provided.
    :param k_range: Range of k explored
    :param silhouette_scores: Average Silhouette Score corresponding to values in k range
    :param optimal_k: The value of k that has the highest average Silhouette Score
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
    if file_path is not None:
        plt.savefig(file_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return


def plot_model_evaluation(forecast_df, model_name, metrics, figsize=(20,13), save_fig=False):
    '''
    Plot model's predictions on training and test sets, along with key performance metrics.
    :param forecast_df: DataFrame consisting of predicted and ground truth consumption values
    :param model_name: model identifier
    :param metrics: key performance metrics
    :param figsize: size of matplotlib figure
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
    forecast_df[pd.isnull(forecast_df["model"])][["gt", "forecast"]].plot(color=["black", "red"], title="Test Set Forecast",
                                                          grid=True, ax=ax2)
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
        plt.savefig(cfg['PATHS']['VISUALIZATIONS'] + model_name + '_forecast_' +
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return