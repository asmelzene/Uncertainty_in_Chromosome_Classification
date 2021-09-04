import datetime                  # test + Uncertainty
import pandas as pd              # test + Uncertainty
import numpy as np               # test + Uncertainty
import pickle5 as pickle         # Uncertainty
from tqdm import tqdm

#import test_model as tm                    # test
import test_uncertainty_Softmax_v4 as tuc   # Uncertainty
from humanfriendly import format_timespan

from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import performance

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from scipy import stats
from scipy.special import entr
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pylab

def run(config_file = '../config/VAL/config_densenet121_SCRATCH.ini', sample_weights = [-1]
                   , u_Dataset = 'arkusai', idx2class = '../data/index_to_class.csv'
                   , WL_max = 5, FR_min = 0.5, FR_max = 1.0, step_size1 = 1, step_size2 = 0.01, plus_minus = 5
                   , save_results=True, save_figure=True, fig_types=['catplot_box', 'catplot_boxen', 'hist_old', 'hist_prob', 
                    'jointplot_correct', 'jointplot_hue', 'jointplot_incorrect', 'jointplot_kde']):
    p_config_file = config_file.split('/')
    p_config_file2 = p_config_file[3].replace('squeezenet1_0', 'squeezenet10').split('_')
    p_config_file3 = p_config_file2[2].split('.')    
    u_Data = p_config_file[2]
    u_Model = p_config_file2[1]
    u_type = p_config_file3[0]
        
    use_sample_weights = sample_weights
    tuc1 = tuc.test_uncertainty(config_file, use_sample_weights)

    index_to_class = pd.read_csv(idx2class)
    class_to_index = index_to_class.set_index('class')
    class_to_index.columns=['idx']

    lst_acc_models, lst_column = tuc1.accuracy_per_model(use_sample_weights)        

    acc_res = tuc1.run()
    #acc_res = (accuracy, c_unmatch, c_2nd_match, lst_worst_acc)  # c=count
    df_results_eval = tuc1.df_uncertainty  # df_final
    df_final_summary = tuc1.df_final_summary

    l_column = tuc1.enhance_evaluations(lst_column)

    df_results_eval['v_entropy'] = tuc1.df_result_ensemble['v_entropy']
    df_results_eval['preds_rate'] = tuc1.df_result_ensemble['preds_rate']

    tuc1.min_max_norm(df_results_eval, metric='var')
    df_results_eval.rename(columns={'u_entr': 'u_entr1'}, inplace=True)
    tuc1.min_max_norm(df_results_eval, metric='entr')
    tuc1.min_max_norm(df_results_eval, metric='var')

    df_ensemble_res, df_flexible = tuc1.vertical_check(lst_column, upper_th=FR_max, lower_th=FR_min)

    #model_similarities(n_models)
    ##################################  PLOT the FIGURES  ######################################################
    n_ensemble = len(lst_acc_models)
    save_name = f'../figures/figure_{u_Dataset}_{u_Model}_{u_Data}_{u_type}_E{n_ensemble}_{WL_max}_{FR_min}_{FR_max}'
    
    if save_results:
        plt.close('all')
        fig1, ((plt_sub1, plt_sub2), (plt_sub3, plt_sub4)) = plt.subplots(nrows=2, ncols=2, figsize=(15,8))
        #fig2, ((plt_sub12, plt_sub22), (plt_sub32, plt_sub42)) = plt.subplots(nrows=2, ncols=2, figsize=(15,8))
        df_Softmax = tuc1.plot_metrics_test(plt_m=plt_sub1, title='Softmax Score', metric_name='best_score', 
                                step_size=step_size2, is_norm=False, plus_minus = plus_minus, plot_graph = True)
        df_Top2 = tuc1.plot_metrics_test(plt_m=plt_sub2, title='Top-2 Difference', metric_name='Certainty', step_size=step_size1, is_norm=False, plus_minus = plus_minus, plot_graph = True)
        df_uncertainty = tuc1.plot_metrics_test(plt_m=None, title='Top-2_Uncertainty', metric_name='uncertainty', 
                        step_size=step_size2, higher_better=False, is_norm=False, plus_minus = plus_minus, plot_graph = False)

        ######  ************************** #######
        df_Entropy_H = tuc1.plot_metrics_test(plt_m=plt_sub3, title='Entropy of Ensemble-Softmax [sum of entr(means)]', metric_name='u_entr', step_size=step_size2, higher_better=False, is_norm=False, plus_minus = plus_minus, 
                                              plot_graph = True)

        df_Var_H = tuc1.plot_metrics_test(plt_m=plt_sub4, title='u_var', metric_name='u_var', step_size=step_size2, 
                               higher_better=True, is_norm=False, plus_minus = plus_minus, plot_graph = True)
        
        fig1.tight_layout()
        fig1.savefig(f'{save_name}.png')
    
#         plt.tight_layout()
#         plt.savefig(f'{save_name}.png')
    
    def histogram_OLD(data_1, data_2, style='darkgrid', kde=True, bins=10):
        sns.set_style(style)
        sns.distplot(data_1, kde=kde, bins=bins)
        sns_plot = sns.distplot(data_2, kde=kde, bins=bins)

        return sns_plot
    
    if save_figure:                
        df_results_eval_test=df_results_eval.loc[:,['Certainty','uncertainty', 'Incorrect', 
                                                'best_score', 'u_entr', 'u_var']]
        df_results_eval_test = df_results_eval_test.astype(float)
        
        metrics = ['Certainty','uncertainty', 'best_score', 'u_entr', 'u_var']
        
        for metric in metrics:
            print(f'********** {metric} ***********')
            plt.close('all')
            d_match = df_results_eval[df_results_eval['actual_class'] == df_results_eval['best_pred']][metric]
            d_unmatch = df_results_eval[df_results_eval['actual_class'] != df_results_eval['best_pred']][metric]
            # histogram_OLD(d_match, d_unmatch)
            
            if 'hist_old' in fig_types:
                print(f'********** hist_old ***********')
                plt.close('all')
                sns_plot1 = histogram_OLD(d_match, d_unmatch)
                sns_plot1.figure.savefig(f'{save_name}_{metric}_hist_old.png')
                        
            if 'hist_prob' in fig_types:
                print(f'********** hist_prob ***********')
                plt.close('all')
                sns_plot2 = sns.displot(df_results_eval_test, x=metric, hue='Incorrect', stat='probability',
                                common_norm=False, aspect=2, height=5, bins=10)
                sns_plot2.savefig(f'{save_name}_{metric}_hist_prob.png')
                        
            if 'catplot_boxen' in fig_types:
                print(f'********** catplot_boxen ***********')
                plt.close('all')
                sns_plot3 = sns.catplot(x='Incorrect', y=metric, kind='boxen',
                                       data=df_results_eval_test.sort_values('Incorrect'))
                sns_plot3.savefig(f'{save_name}_{metric}_catplot_boxen.png')
            
            if metric != 'best_score':
                if 'jointplot_incorrect' in fig_types:
                    print(f'********** jointplot_incorrect ***********')
                    plt.close('all')
                    sns_plot4 = sns.jointplot(data=df_results_eval_test[df_results_eval_test['Incorrect']==1], 
                                  x=metric, y="best_score", marker="+", s=50, marginal_kws=dict(bins=10, fill=False))
                    sns_plot4.savefig(f'{save_name}_{metric}_jointplot_incorrect.png')
            
                if 'jointplot_correct' in fig_types:
                    print(f'********** jointplot_correct ***********')
                    plt.close('all')
                    sns_plot5 = sns.jointplot(data=df_results_eval_test[df_results_eval_test['Incorrect']==0], 
                                  x=metric, y="best_score", marker="+", s=50, marginal_kws=dict(bins=10, fill=False))
                    sns_plot5.savefig(f'{save_name}_{metric}_jointplot_correct.png')
            
                if 'jointplot_kde' in fig_types:
                    print(f'********** jointplot_kde ***********')
                    plt.close('all')
                    sns_plot6 = sns.jointplot(data=df_results_eval_test, x=metric, y='best_score', hue='Incorrect',
                                     kind='kde', common_norm=False)
                    sns_plot6.savefig(f'{save_name}_{metric}_jointplot_kde.png')
            
                if 'jointplot_hue' in fig_types:
                    print(f'********** jointplot_hue ***********')
                    plt.close('all')
                    sns_plot7 = sns.jointplot(data=df_results_eval_test, x=metric, y='best_score', hue='Incorrect')
                    sns_plot7.savefig(f'{save_name}_{metric}_jointplot_hue.png')
                
                if 'catplot_box' in fig_types:
                    print(f'********** catplot_box ***********')
                    plt.close('all')
                    sns_plot8 = sns.catplot(x='best_score', y=metric, row='Incorrect', kind='box',
                                    orient='v', height=4.5, aspect=4, data=df_results_eval_test)
                    sns_plot8.savefig(f'{save_name}_{metric}_catplot_box.png')
    
        #     fig2.tight_layout()
        #     fig2.savefig(f'{save_name}_type2.png')
    
    if save_results:
        ##################################  RANGE CALCULATIONS  ####################################################
        ###### df_range_Softmax
        df_range_Softmax = tuc1.tolerance_range(df_Softmax, FR_min=FR_min, FR_max=FR_max, WL_max = WL_max)

        ###### df_range_Top2 certainty
        df_range_Top2 = tuc1.tolerance_range(df_Top2, FR_min=FR_min, FR_max=FR_max, WL_max = WL_max)

        ###### df_range_Top2 UNcertainty
        df_range_uncertainty = tuc1.tolerance_range(df_uncertainty, FR_min=FR_min, FR_max=FR_max, WL_max = WL_max)

        ###### df_Entropy_H[df_Entropy_H['fail_rate']>=0.45]
        df_range_Entropy_H = tuc1.tolerance_range(df_Entropy_H, FR_min=FR_min, FR_max=FR_max, WL_max = WL_max)
        ###### df_range_Entropy_H

        ######df_Var_H[df_Var_H['fail_rate']<0.5]
        df_range_Var_H = tuc1.tolerance_range(df_Var_H, FR_min=FR_min, FR_max=FR_max, WL_max = WL_max)
        ###### df_range_Var_H

        ##################################  MERGE RESULTS  ########################################################

        model = df_results_eval.loc[0, 'model']
        #parse_metrics(model, acc_res, df_metric, df_range_metric, KPI='fail_rate', th_KPI = 0.5, is_smaller = True)
        df_Report_Softmax = tuc1.parse_metrics(model, acc_res, df_Softmax, df_range_Softmax)
        df_Report_Top2 = tuc1.parse_metrics(model, acc_res, df_Top2, df_range_Top2)
        df_Report_Uncertainty = tuc1.parse_metrics(model, acc_res, df_uncertainty, df_range_uncertainty)
        df_Report_Entr = tuc1.parse_metrics(model, acc_res, df_Entropy_H, df_range_Entropy_H)
        df_Report_Var = tuc1.parse_metrics(model, acc_res, df_Var_H, df_range_Var_H)

        dict_uncertainty = tuc1.find_mean(df_results_eval, metric = 'uncertainty') # [0:100] lower better ... uncertainty: [0: 1]
        dict_Certainty = tuc1.find_mean(df_results_eval, metric = 'Certainty')    # [0:100] higher better ... uncertainty: [0: 1]
        dict_best_score = tuc1.find_mean(df_results_eval, metric = 'best_score') # [0:100] higher better = Softmax
        dict_u_entr = tuc1.find_mean(df_results_eval, metric = 'u_entr')        # [0:1]   lower better  = normalized Entropy: e/e_max
        dict_u_var = tuc1.find_mean(df_results_eval, metric = 'u_var')         # [0:1]   higher better = normalized_min_max

        dict_metrics = {'Softmax': dict_best_score, 'Certainty': dict_Certainty, 'uncertainty': dict_uncertainty
                        , 'Entropy': dict_u_entr, 'Variance': dict_u_var}

        df_Report_Softmax = tuc1.parse_metrics(model, acc_res, df_Softmax, df_range_Softmax)
        df_Report_Top2 = tuc1.parse_metrics(model, acc_res, df_Top2, df_range_Top2)
        df_Report_uncertainty = tuc1.parse_metrics(model, acc_res, df_uncertainty, df_range_uncertainty, KPI='fail_rate', 
                                        th_KPI = FR_min, is_smaller = False)
        df_Report_Entr = tuc1.parse_metrics(model, acc_res, df_Entropy_H, df_range_Entropy_H, KPI='fail_rate', 
                                        th_KPI = FR_min, is_smaller = False)
        df_Report_Var = tuc1.parse_metrics(model, acc_res, df_Var_H, df_range_Var_H)
        dict_reports = {'Softmax': df_Report_Softmax, 'Certainty': df_Report_Top2, 'uncertainty': df_Report_uncertainty
                        , 'Entropy': df_Report_Entr, 'Variance': df_Report_Var}

        col_names =  ['DataSet', 'Model', 'Data', 'Accuracy', 'Metric', 'Fail_Rate', 'Work_Load_P', 'Value'
                      , 'Tolerance_Range', 'Mean_ALL', 'Mean_Correct', 'Mean_Incorrect', 'Acc_Per_Model'
                      , 'Worst_Acc_Per_Class']
        df_Summary  = pd.DataFrame(columns = col_names)

        # u_Dataset, u_Model, u_Data
        # Softmax, Certainty, uncertainty, Entropy, Variance
        for i, metric in enumerate(dict_metrics.keys()):
            temp_row = [u_Dataset, u_Model, u_Data, acc_res, metric, dict_reports[metric]['fail_rate_m'][0]
                        , dict_reports[metric]['work_load_m'][0], dict_reports[metric]['val_m'][0]
                        , f"{dict_reports[metric]['range_len'][0]} ({dict_reports[metric]['range_min'][0]}-\
        {dict_reports[metric]['range_max'][0]})"
                , dict_metrics[metric]['mean_all'], dict_metrics[metric]['mean_correct'], dict_metrics[metric]['mean_incorrect']
                , lst_acc_models,  dict_reports[metric]['worst_acc'][0]]

            df_Summary.loc[i] = temp_row
                
    
        save_name=f'../uncertainty/uncertainty_{u_Dataset}_{u_Model}_{u_Data}_{u_type}_E{n_ensemble}_{WL_max}_{FR_min}_{FR_max}'
        df_Summary.to_csv(f'{save_name}.csv')

        a_file = open(f'{save_name}.pkl', 'wb')
        pickle. dump(tuc1, a_file)
        a_file.close()
    
        
        
        