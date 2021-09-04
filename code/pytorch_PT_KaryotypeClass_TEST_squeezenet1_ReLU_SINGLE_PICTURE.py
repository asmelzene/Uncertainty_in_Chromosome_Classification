import datetime                  # test + Uncertainty
import pandas as pd              # test + Uncertainty
import numpy as np               # test + Uncertainty
import pickle                    # Uncertainty

import test_model_single_picture as tm          # test
import test_uncertainty_ReLU as tuc   # Uncertainty
from humanfriendly import format_timespan

start_date=datetime.datetime.now()
print(f'start_date {start_date}')

tm1 = tm.test_model()

#tm1.data_dir = 'data/BioImLab_single_chromosomes_05.04.2021/test/'
#tm1.model_dir = 'models'
tm1.model_list = {'squeezenet1'} #{'alexnet', 'densenet121', 'resnet18', 'squeezenet1', 'VGG11_bn'}
tm1.model_meta_csv = '/home/melih/arkusai/models/model_meta_squeezenet1.csv' #'model_meta_densenet121.csv'  ## model_meta.csv
tm1.img_dir = '/home/melih/arkusai/data/'
tm1.picture = 'MA160865191500026.005-K_11_0_.png'
#tm1.test_data_csv = 'data/BioImLab_single_chromosomes_05.04.2021/chromosome_data_test.csv'

print(f'model running: {datetime.datetime.now()}')
df_result_ensemble = tm1.run()

pd.set_option('max_colwidth', -1)
df_result_ensemble

# tm1.df_result
print(f'uncertainty running: {datetime.datetime.now()}')

df_result_ensemble.to_pickle('/home/melih/arkusai/results/df_result_ensemble2_new.pkl')

with open('/home/melih/arkusai/results/df_result_ensemble2_new.pkl', 'rb') as fp:
     df_result_ensemble = pickle.load(fp)

df_result_ensemble

tuc1 = tuc.test_uncertainty(df_result_ensemble, single_picture=True)

tuc1.run()
df_final = tuc1.df_uncertainty
df_final_summary = tuc1.df_final_summary

pd.set_option('max_colwidth', -1)
pd.set_option('display.max_columns', None)
# print(df_final)

# print(df_final_summary)

end_date=datetime.datetime.now()
print(f'ALL done: {end_date}')

t_delta = end_date - start_date
format_timespan(t_delta.seconds)