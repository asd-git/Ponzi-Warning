# trans_old feature
import numpy as np
import pandas as pd
# root_path = '.././data_new/'
# root_path = '.././data_new/eth/Dy_GRU_data/100/'
root_path = '.././data_no_aug/'

trans_df = pd.read_csv(root_path+'/eth/raw/eth_node_attributes.txt',sep=',',header=None)
trans_arctan = trans_df.apply(lambda x: np.arctan(x) * (2 / np.pi)) #非线性归一化
trans_arctan.to_csv(root_path+'/eth/eth_arctan_node_attributes.txt',sep=',',index=False,header=False)
# code feature
# code_df = pd.read_csv(root_path+'/eth/raw/eth_code_attributes.txt',sep=',',header=None)
# code_arctan = code_df.apply(lambda x: np.arctan(x) * (2 / np.pi)) #非线性归一化
# code_arctan.to_csv(root_path+'/eth/eth_arctan_code_attributes.txt',sep=',',index=False,header=False)

#
# trans_df = pd.read_csv(root_path+'/eth/raw/eth_node_attributes.txt',sep=',',header=None)
# trans_arctan = trans_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# trans_arctan.to_csv(root_path+'/eth/eth_linear_node_attributes.txt',sep=',',index=False,header=False)


# code feature
# code_df = pd.read_csv(root_path+'/eth/raw/eth_code_attributes.txt',sep=',',header=None)
# code_arctan = code_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# code_arctan.to_csv(root_path+'/eth/eth_linear_code_attributes.txt',sep=',',index=False,header=False)
#
#
# trans_df = pd.read_csv(root_path+'/eth/raw/eth_node_attributes.txt',sep=',',header=None)
# trans_arctan = trans_df.apply(lambda x: (x - np.mean(x)) / np.std(x))  #方差归一化
# trans_arctan.to_csv(root_path+'/eth/eth_std_node_attributes.txt',sep=',',index=False,header=False)

# # code feature
code_df = pd.read_csv(root_path+'/eth/raw/eth_code_attributes.txt',sep=',',header=None)
code_arctan = code_df.apply(lambda x: (x - np.mean(x)) / np.std(x))  #方差归一化
code_arctan.to_csv(root_path+'/eth/eth_std_code_attributes.txt',sep=',',index=False,header=False)