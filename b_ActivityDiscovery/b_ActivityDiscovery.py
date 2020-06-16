# SOM
import inspect
import logging
import sys

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn_som.som import SOM
from hyperopt import fmin
from functools import partial

import z_setting_parameters as settings
from u_utils import u_helper as helper, u_utils as utils


def choose_and_perform_clustering_method(clustering_method,
                                         trace_data_without_case_number,
                                         hyp_number_of_clusters):
    """
    This method manages the different clustering methods and starts the selected method. The results of the different
    clustering methods is in the same format.

    @param clustering_method:               Specifying the clustering method that is used
    @param hyp_number_of_clusters:              Specifying the number of clusters. (not used for "k-Means-Elbow" and
                                            "k-Medoids-Elbow")
    @param trace_data_without_case_number:  List of all vectors that should be clustered

    @return:                                result list returns cluster for each vector
    """

    # clustering with the self organizing map by sklearn
    if clustering_method == 'sklearn-SOM':
        # find all possible som dimensions
        # divisor_pairs = utils.find_divisor_pairs(number=number_of_clusters)
        divisor_pairs = utils.find_divisor_pairs(number=hyp_number_of_clusters)
        # only keep the most "squared" shape
        som_dimensions = divisor_pairs[int((divisor_pairs.__len__() - 1) / 2)]

        # perform process model discovery for different parameter combinations and find the best outcome
        space = helper.create_som_param_opt_space()
        # helper variables
        choose_and_perform_clustering_method.predictions = []
        choose_and_perform_clustering_method.inertia = sys.maxsize
        # find the best matching SOM (hyperparameter optimization) with fixed dimensions
        fmin(fn=partial(create_som_with_sklearn, m=som_dimensions[0], n=som_dimensions[1],
                        trace_data_without_case_number=trace_data_without_case_number),
             space=space,
             algo=settings.som_opt_algorithm,
             max_evals=settings.som_opt_attempts,
             verbose=False)
        clustering_result = choose_and_perform_clustering_method.predictions

        # logger
        number_of_clusters = max(clustering_result) + 1
        logger = logging.getLogger(inspect.stack()[0][3])
        logger.setLevel(settings.logging_level)
        logger.info("Clustered data into %s clusters using a %sx%s sklearn SOM.", number_of_clusters, som_dimensions[0],
                    som_dimensions[1])

    # uses K-Means form scikit-learn for clustering the given data
    elif clustering_method == 'k-Means':
        # cluster for each vector, average distance to centroid
        clustering_result = KMeans(n_clusters=hyp_number_of_clusters).fit(trace_data_without_case_number.values).labels_

    # clustering with k-medoids form sk-learn-extra
    elif clustering_method == 'k-Medoids':
        clustering_result = KMedoids(n_clusters=hyp_number_of_clusters, init='k-medoids++').fit(
            trace_data_without_case_number).labels_

    else:
        logger = logging.getLogger(inspect.stack()[0][3])
        logger.setLevel(settings.logging_level)
        error_msg = "'" + clustering_method + "' is not a valid clustering method. Please check the settings."
        logger.error(error_msg)
        raise ValueError(error_msg)

    return clustering_result


def create_som_with_sklearn(params, m, n, trace_data_without_case_number):
    """
    Creates a self organizing map for given dimensions, parameters and data.
    :param params: hyperopt parameters to find the best SOM for given data
    :param m: vertical dimension of the som
    :param n: horizontal dimension of the som
    :param trace_data_without_case_number: the data the SOM is build on
    :return: the inertia of the current som
    """
    # creates a SOM
    sklearn_som = SOM(m=m, n=n, lr=params['lr'], sigma=params['sigma'], dim=trace_data_without_case_number.shape[1])
    # "adapts" the SOM to the data
    sklearn_som.fit(trace_data_without_case_number.values)
    # find the best performing SOM
    if sklearn_som.inertia_ < choose_and_perform_clustering_method.inertia:
        choose_and_perform_clustering_method.predictions = sklearn_som.predict(trace_data_without_case_number.values)
        choose_and_perform_clustering_method.inertia = sklearn_som.inertia_

    return sklearn_som.inertia_


# k-means Vanilla
def custom_kmeans(data, number_of_clusters):
    np_data_array = data.values
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(np_data_array)
    return kmeans.labels_
