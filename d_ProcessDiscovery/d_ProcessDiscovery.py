import inspect
import logging
import os
import re
from scipy import stats
import pandas as pd
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.evaluation.precision import evaluator as precision_evaluator
from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.petri.exporter import exporter as pnml_exporter
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.statistics.end_activities.log import get as ea_get
from pm4py.statistics.start_activities.log import get as sa_get
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.visualization.petrinet import visualizer as pn_visualizer

import z_setting_parameters as settings
from e_Evaluation import e_entropia as entropia


def create_activity_models(output_case_traces_cluster, path_data_sources, dir_runtime_files, dir_dfg_cluster_files,
                           filename_dfg_cluster, rel_proportion_dfg_threshold):
    """
    Creates directly follows graphs out of an event log.
    :param output_case_traces_cluster: traces that are visualised
    :param path_data_sources: path of sources and outputs
    :param dir_runtime_files: folder containing files read and written during runtime
    :param dir_dfg_cluster_files: folder containing dfg png files
    :param filename_dfg_cluster: filename of dfg file (per cluster)
    :param rel_proportion_dfg_threshold: threshold for filtering out sensors in dfg relative to max occurrences of a sensor
    :return:
    """

    # keep only needed columns
    output_case_traces_cluster = output_case_traces_cluster.reindex(
        columns={'Case', 'LC_Activity', 'Timestamp', 'Cluster'})

    # rename columns so pm4py understands the columns
    output_case_traces_cluster = output_case_traces_cluster.rename(
        columns={'Case': 'case:concept:name',
                 'LC_Activity': 'concept:name',
                 'Timestamp': 'time:timestamp'})

    # create directory for dfg pngs
    os.mkdir(path_data_sources + dir_runtime_files + dir_dfg_cluster_files)
    # create dfg for each cluster
    clusters = output_case_traces_cluster.Cluster.unique()
    for cluster in clusters:
        log = output_case_traces_cluster.loc[output_case_traces_cluster.Cluster == cluster]
        log = log.astype(str)
        log['time:timestamp'] = pd.to_datetime(output_case_traces_cluster['time:timestamp'])
        # convert pandas data frame to pm4py event log for further processing
        log = log_converter.apply(log)

        # keep only activities with more than certain number of occurrences
        activities = attributes_get.get_attribute_values(log, 'concept:name')
        # determine that number relative to the max number of occurrences of a sensor in a cluster.
        # (the result is the threshold at which an activity/activity strand is kept)
        min_number_of_occurrences = round((max(activities.values()) * rel_proportion_dfg_threshold), 0)
        activities = {x: y for x, y in activities.items() if y >= min_number_of_occurrences}
        log = attributes_filter.apply(log, activities)

        # create png dfg file
        export_dfg_imagefile(log=log,
                             path_data_sources=path_data_sources,
                             dir_runtime_files=dir_runtime_files,
                             dir_dfg_files=dir_dfg_cluster_files,
                             filename_dfg=filename_dfg_cluster.format(cluster=str(cluster)))

    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(settings.logging_level)
    logger.info("Saved directly follows graphs for each cluster into '../%s'.",
                path_data_sources + dir_runtime_files + dir_dfg_cluster_files)


def create_process_model(output_case_traces_cluster, dir_runtime_files):
    path_data_sources = settings.path_data_sources
    filename_log_export = settings.filename_log_export
    dir_petri_net_files = settings.dir_petri_net_files
    filename_petri_net = settings.filename_petri_net
    filename_petri_net_image = settings.filename_petri_net_image
    dir_dfg_files = settings.dir_dfg_files
    filename_dfg = settings.filename_dfg
    miner_type = settings.miner_type
    metric_to_be_maximised = settings.metric_to_be_maximised

    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(settings.logging_level)

    # export routine/cluster list for Disco import etc.
    # only keep one row with the same cluster for the same day
    # sort dataframe
    output_case_traces_cluster_sorted = output_case_traces_cluster.sort_values(by=['Case', 'Timestamp'])
    # define columns to compare
    cols = ["Routine", "Cluster"]
    output_routine_cluster_export = output_case_traces_cluster.loc[
        (output_case_traces_cluster_sorted[cols].shift() != output_case_traces_cluster_sorted[cols]).any(axis=1)]
    # add date column to be used as 'CaseID' later
    output_routine_cluster_export['Date'] = output_routine_cluster_export['Timestamp'].dt.date
    # save as csv
    output_routine_cluster_export.to_csv(
        path_or_buf=settings.path_data_sources + dir_runtime_files + 'Clusters_in_Routines.csv',
        index=False,
        sep=';')

    # go through all routines (morning/noon/evening/..)
    routine_list = output_case_traces_cluster['Routine'].unique().tolist()
    # routine_metrics = pd.DataFrame(columns=['Quantity', metric_to_be_maximised], index=[routine_list])
    routine_metrics = pd.DataFrame(columns=['Quantity', 'Precision', 'Fitness', 'F1'], index=[routine_list])
    for routine in routine_list:
        routine_log = output_case_traces_cluster.loc[output_case_traces_cluster['Routine'] == routine]

        # save length of log in order to calculate a weighted average for the metric
        routine_metrics.at[routine, 'Quantity'] = len(routine_log)

        # create a log that can be understood by pm4py
        pm4py_log = \
            convert_log_to_pm4py(log=routine_log)

        # export log as XES-file
        xes_exporter.apply(pm4py_log, path_data_sources + dir_runtime_files + routine + '-' + filename_log_export)
        logger.info("Exported log export into '../%s'.",
                    path_data_sources + dir_runtime_files + routine + '-' + filename_log_export)

        # create png dfg file
        export_dfg_imagefile(log=pm4py_log,
                             path_data_sources=path_data_sources,
                             dir_runtime_files=dir_runtime_files,
                             dir_dfg_files=dir_dfg_files,
                             filename_dfg=routine + '-' + filename_dfg)
        logger.info("Saved directly follows graph into '../%s'.",
                    path_data_sources + dir_runtime_files + dir_dfg_files + routine + '-' + filename_dfg)

        # apply miner, export petri-nets and return selected metric (precision, fitness, ...)
        metrics = apply_miner(log=pm4py_log,
                              path_data_sources=path_data_sources,
                              dir_runtime_files=dir_runtime_files,
                              dir_petri_net_files=dir_petri_net_files + routine + '/',
                              filename_petri_net=routine + '-' + filename_petri_net,
                              filename_petri_net_image=routine + '-' + filename_petri_net_image,
                              filename_log_export=routine + '-' + filename_log_export,
                              miner_type=miner_type,
                              metric_to_be_maximised=metric_to_be_maximised)
        routine_metrics.at[routine, metrics] = metrics.values()

    weighted_metric_average = {}
    weighted_metric_average['Precision'] = (routine_metrics['Quantity'] *
                                            routine_metrics['Precision']).sum() / routine_metrics['Quantity'].sum()
    weighted_metric_average['Fitness'] = (routine_metrics['Quantity'] *
                                          routine_metrics['Fitness']).sum() / routine_metrics['Quantity'].sum()
    weighted_metric_average['F1'] = (routine_metrics['Quantity'] *
                                     routine_metrics['F1']).sum() / routine_metrics['Quantity'].sum()

    # weighted_metric_average = \
    #     (routine_metrics['Quantity'] * routine_metrics[metric_to_be_maximised]).sum() / \
    #     routine_metrics['Quantity'].sum()

    return weighted_metric_average


def convert_log_to_pm4py(log):
    """
    Restructures a pandas log to a pm4py specific log.
    :param log: the event log saved in a pandas data frame
    :return: a log pm4py methods can get applied on
    """

    log['Date'] = log['Timestamp'].dt.date

    log = log.reindex(columns={'Case', 'Timestamp', 'Cluster', 'Date'})
    log_pm4py = log.rename(columns={'Date': 'case:concept:name',
                                    'Cluster': 'concept:name',
                                    'Timestamp': 'time:timestamp'})
    log_pm4py = log_pm4py.astype(str)
    log_pm4py['time:timestamp'] = pd.to_datetime(log_pm4py['time:timestamp'])

    # reduce log entries, so that every case only appears once
    log_pm4py.drop_duplicates(subset=['Case'], inplace=True)

    # reset row index
    log_pm4py = log_pm4py.reset_index(drop=True)

    return log_pm4py


def export_dfg_imagefile(log, path_data_sources, dir_runtime_files, dir_dfg_files, filename_dfg):
    """
    Creates a directly follows graph (dfg) out of the given log.
    :param log: log the dfg is based on
    :param path_data_sources: path of sources and outputs
    :param dir_runtime_files: folder containing files read and written during runtime
    :param dir_dfg_files: folder the dfg image files are saved into
    :param filename_dfg: name of the dfg image file
    :return:
    """

    # create dfg out of event log
    dfg = dfg_discovery.apply(log)

    # define start and end
    start_activities = sa_get.get_start_activities(log)
    end_activities = ea_get.get_end_activities(log)

    # create png of dfg (if the graph does not show a graph, it is possible that the sensors did not trigger often)
    # parameter has to be dfg0, because apply method requires a dfg0 mehtod, maybe depending on the pm4py version?
    gviz = dfg_visualization.apply(dfg0=dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY,
                                   parameters={'start_activities': start_activities,
                                               'end_activities': end_activities})
    # save dfg image to the drive
    dfg_visualization.save(gviz, path_data_sources + dir_runtime_files + dir_dfg_files + filename_dfg)

    return


def apply_miner(log,
                path_data_sources,
                dir_runtime_files,
                dir_petri_net_files,
                filename_petri_net,
                filename_petri_net_image,
                filename_log_export,
                miner_type,
                metric_to_be_maximised):
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(settings.logging_level)

    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}
    log_converted = log_converter.apply(log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

    # Choose miner type:
    if miner_type == 'heuristic':
        logger.info("Applying heuristic miner to log.")
        net, initial_marking, final_marking = heuristics_miner.apply(log_converted, parameters={
            heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.8})
        logger.info("Done with heuristic miner.")
    elif miner_type == 'inductive':
        logger.info("Applying inductive miner to log.")
        net, initial_marking, final_marking = inductive_miner.apply(log_converted)
        logger.info("Done with inductive miner.")
    else:
        return None

    # create directory for petri net files
    os.makedirs(path_data_sources + dir_runtime_files + dir_petri_net_files)

    # export petri net png
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.save(gviz, path_data_sources + dir_runtime_files + dir_petri_net_files + filename_petri_net_image)
    logger.info("Exported petri net image file into '../%s'.",
                path_data_sources + dir_runtime_files + dir_petri_net_files + filename_petri_net_image)

    # export petri net pnml
    pnml_exporter.apply(net, initial_marking,
                        path_data_sources + dir_runtime_files + dir_petri_net_files + filename_petri_net,
                        final_marking=final_marking)
    logger.info("Exported petri net pnml file into '../%s'.",
                path_data_sources + dir_runtime_files + dir_petri_net_files + filename_petri_net)

    # logger
    logger.info("Calculating %s.", metric_to_be_maximised)
    # Choose target value to be maximised
    metrics = {}
    if metric_to_be_maximised == 'Fitness':
        precision = replay_fitness_evaluator.apply(log_converted, net, initial_marking, final_marking,
                                                   variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
        metrics['Fitness'] = precision['log_fitness']
    elif metric_to_be_maximised == 'Precision':
        # metrics = precision_evaluator.apply(log, net, initial_marking, final_marking,
        #                                    variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
        metrics['Precision'] = precision_evaluator.apply(log, net, initial_marking, final_marking,
                                                         variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    elif metric_to_be_maximised == 'F1':
        fitness = replay_fitness_evaluator.apply(log_converted, net, initial_marking, final_marking,
                                                 variant=replay_fitness_evaluator.Variants.TOKEN_BASED)

        metrics['Precision'] = precision_evaluator.apply(log, net, initial_marking, final_marking,
                                                         variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
        metrics['Fitness'] = fitness['log_fitness']

        metrics['F1'] = stats.hmean([metrics['Precision'], metrics['Fitness']])
    else:
        # evaluate precision/fitness with entropia
        # assemble paths for entropia evaluation
        xes_file = path_data_sources + dir_runtime_files + filename_log_export
        pnml_file = path_data_sources + dir_runtime_files + settings.dir_petri_net_files + filename_petri_net
        entropia_res = None
        if metric_to_be_maximised == 'entropia:Precision':
            entropia_res = entropia.compute_entropia("pmp", xes_file=xes_file, pnml_file=pnml_file)
        elif metric_to_be_maximised == 'entropia:Fitness':
            entropia_res = entropia.compute_entropia("pmr", xes_file=xes_file, pnml_file=pnml_file)

        if 'Exception' in entropia_res or 'NaN' in entropia_res:

            logger.info("%s could not get calculated. The result is set to 0. The error was: %s",
                        metric_to_be_maximised, entropia_res)
            metrics = 0
        else:
            # the result is a number
            # only keep the computed number of the entropia result and cut every other char

            metrics[metric_to_be_maximised] = float(re.sub('[^\d.]', '', entropia_res))

    logger.info("%s calculated: %s", metric_to_be_maximised, metrics)

    # alternative metrics
    # generalization = generalization_evaluator.apply(log, net, im, fm)
    # simplicity = simplicity_evaluator.apply(net)

    return metrics
