import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime

# Set some matplotlib parameters
mpl.rcParams['figure.figsize'] = (12, 10)


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