import os
import re

import pandas as pd
import numpy as np

from sklearn import preprocessing

from collections import defaultdict


def scale_test(df= None, mean_dict= None, std_dict= None):
	for col in df.columns:
		df[col]= (df[col]- mean_dict[col])/std_dict[col]

	return df


def normalize(df= None):
	for col in df.columns:
		df[col]= (df[col]-df[col].min())/(df[col].max() - df[col].min())

	return df


def scale(df= None, ddof= 0):
	mean_dict= dict()
	std_dict= dict()

	for col in df.columns:
		mean_dict[col]= df[col].mean()
		std_dict[col]= df[col].std()

		df[col]= (df[col]-df[col].mean())/df[col].std()

	return df, mean_dict, std_dict


def std_scalar(df= None, label= []):
	labs= df[label]

	df.drop(labs.columns, inplace= True, axis= 1)
	cols= df.columns
	indx= df.index

	df_scaled= pd.DataFrame(preprocessing.scale(df.values), columns= cols, index= indx)

	ddf= pd.merge(df_scaled, labs, on= indx)
	ddf.index= indx
	ddf.drop(['key_0'], inplace= True, axis=1)
	return ddf


def process_mps_st_dfs(p1= None, p2= None):

    # Processing the dataframes
    def process_df_for_training(df= pd.DataFrame()):

        # Functions to convert the dataframe's string arrays to ints and/or floats
        def float_convert_list(i, splitter= ' '):
            if splitter == ' ':
                a= [float(j) for j in i[1:-2].split()]
            elif splitter == ',':
                a= [float(j) for j in i[1:-2].split(',')]
            else:
                a= [float(j) for j in i[1:-2].split()]
            return a


        def convertTo_floatLists(x, splitter= None):
            a= []
            b= []
            c= []

            for i in x:
                entry= float_convert_list(i, splitter= splitter)
                a.append(entry[0])
                b.append(entry[1])
                c.append(entry[2])

            return a, b, c


        def convertTo_floatLists_from4by3(x):
            X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4= [], [], [], [], [], [], [], [], [], [], [], []

            pattern= r'[^\w.-]'
            for i in x:
                j= [float(k) for k in re.sub(pattern, ' ', i).split()]

                X1.append(j[0])
                Y1.append(j[1])
                Z1.append(j[2])

                X2.append(j[3])
                Y2.append(j[4])
                Z2.append(j[5])

                X3.append(j[6])
                Y3.append(j[7])
                Z3.append(j[8])

                X4.append(j[9])
                Y4.append(j[10])
                Z4.append(j[11])

            return X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4

        # Splitting 3 columns into 9, converting their entries to float and saving them
        eC0_1, eC0_2, eC0_3= convertTo_floatLists(df.elCents0, splitter= ' ')
        eC1_1, eC1_2, eC1_3= convertTo_floatLists(df.elCents1, splitter= ' ')

        # old
        # eC_diff_1, eC_diff_2, eC_diff_3= np.array(eC1_1)- np.array(eC0_1), np.array(eC1_2)- np.array(eC0_2), np.array(eC1_3)- np.array(eC0_3)
        #new
        eC_diff_1, eC_diff_2, eC_diff_3= convertTo_floatLists(df.uFigCents, splitter= ',')

        nDC_x, nDC_y, nDC_z= convertTo_floatLists(df.normDirCos, splitter= ',')

        ### Splitting uFigNodes/Disp into 12 entries from a (4, 3) array
        #old
        # X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4= convertTo_floatLists_from4by3(df.figDisp)
        #new
        X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4= convertTo_floatLists_from4by3(df.uFigNodes)

        # Creating a dictionary of newly created variables, followed by mps_df dataframe
        d= {'elCents0_1': eC0_1,
           'elCents0_2': eC0_2,
           'elCents0_3': eC0_3,
           'elCents1_1': eC1_1,
           'elCents1_2': eC1_2,
           'elCents1_3': eC1_3,
            'elCents1_diff': eC_diff_1,
            'elCents2_diff': eC_diff_2,
            'elCents3_diff': eC_diff_3,
           'nDCos_x': nDC_x,
           'nDCos_y': nDC_y,
           'nDCos_z': nDC_z,
           'X1f': X1,
           'Y1f': Y1,
           'Z1f': Z1,
           'X2f': X2,
           'Y2f': Y2,
           'Z2f': Z2,
           'X3f': X3,
           'Y3f': Y3,
           'Z3f': Z3,
           'X4f': X4,
           'Y4f': Y4,
           'Z4f': Z4}

        new_df= pd.DataFrame(d)

        new_df['elemId']= df.elemId.values
        new_df['elConnect']= df.elConnect.values
        new_df['temp']= df.temp.values
        new_df['arGrowth']= df.arGrowth.values
        new_df['maxPrin(%)']= df['maxPrin(%)'].values
        new_df['sThin(%)']= df['sThin(%)'].values

        # Dropping some columns and their engineered counter parts exist in the dataframe
        new_df.drop(['elCents0_1', 'elCents0_2', 'elCents0_3', 'elCents1_1', 'elCents1_2', 'elCents1_3'], axis= 1, inplace= True)

        # Setting index
        new_df.set_index(['elemId', 'elConnect'], inplace= True)


        return new_df


    # Importing dfs
    df1= pd.read_csv(p1)
    df2= pd.read_csv(p2)
    
    df= pd.merge(df1, df2, on= ['elemId', 'elConnect'])
    
    return process_df_for_training(df= df)
    