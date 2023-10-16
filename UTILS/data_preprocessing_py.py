import numpy as np
import pandas as pd
import datetime 

def pad_nans(df):
    for j, col in enumerate(df.columns): 
        for i in range(df.shape[0] ):
            if pd.isna(df.iloc[i,j]):
                #df.iloc[i,j] = (df.iloc[i-1,j]+df.iloc[i+1,j])/2
                df.iloc[i,j] = (df.iloc[i-1,j])
#                print(j, i, col, df.iloc[i,j])
    return df;

def set_timestamp_as_index_old(source_df, format='%Y/%m/%d %H:%M:%S'):
    target_df = source_df.iloc[:, 2:];
    df_ts=list(map(lambda x: x.split('.')[0], source_df['Timestamp'].values));
    df_ts=pd.to_datetime(df_ts, format=format);
    target_df.index = df_ts;
    return target_df;
    
def set_timestamp_as_index_new(source_df, ts_format = '%Y/%m/%d %H:%M:%S.%f'):
    target_df = source_df.iloc[:, 2:];
    # df_ts=list(map(lambda x: x.split('.')[0], source_df['Timestamp'].values));
    
    target_df.index = pd.to_datetime(source_df['Timestamp'], format=ts_format);
    return target_df;

def build_histogram(column_values, bins_count):
    min_val = np.min(column_values);
    max_val = np.max(column_values); 
    step = (max_val - min_val) / bins_count;
    bins = np.arange(min_val, max_val, step);
    return np.histogram(column_values, bins);

def determine_data_range(column_values, valid_data_percent = 95):
    bins_count = 100;
    samples_count = len(column_values);
    hist, bins = build_histogram(column_values, bins_count);
    
    median = np.median(column_values);
    min_id = 0; 
    max_id = -1;
    min_val = bins[0];
    max_val = bins[-1]; 
    non_valid_points = 0;
    non_valid_points_max_count = samples_count * (100 - valid_data_percent) / 100;
    
    while(non_valid_points < non_valid_points_max_count):
        med_max = max_val - median;
        min_med = median - min_val;
        if(med_max < min_med):
            non_valid_points += hist[min_id];
            min_id += 1;
            min_val = bins[min_id];
        else:
            non_valid_points += hist[max_id];
            max_id -= 1;
            max_val = bins[max_id];
    return min_val, max_val, hist, bins;
    
def drop_peaks(column_values, min_val, max_val):
    result_values = []
    for row, value in enumerate(column_values):
        if (value < min_val) or (value > max_val):
            result_values.append(np.NaN)
            print('replasing row', row, ' value', value)    
        else: 
            result_values.append(value)
    return result_values

def get_stationary_variance(df_col, window_size):
    mean = np.convolve(df_col, np.ones(window_size)/window_size, mode="same");
    noise = (df_col - mean);
    variance = np.sqrt(np.power(noise[window_size:-window_size], 2).mean());
    return variance, mean, noise

def get_variance(df_col):
    mean = df_col.values.mean();
    noise = (df_col - mean);
    variance = np.sqrt(np.power(noise, 2).mean());
    return variance, mean, noise

def group_collect_statistics(dfs_groups, format='%Y/%m/%d %H:%M:%S'):
    df_statistics = pd.DataFrame();
    row = 0;
    for k, v in dfs_groups.items():
        # l = pd.concat(map(get_single_segment_statistics, v, format = format))
        l = pd.concat(map(lambda x : get_single_segment_statistics(x, format = format), v))
        seg = v[0]['Current segment'].values[0];
        df_statistics.loc[row,'Segment'] = seg;
        df_statistics.loc[row,'Duration'] = l.Duration.mean();
        df_statistics.loc[row,'Duration Variance'] = get_variance(l.Duration)[1];
        df_statistics.loc[row,'Samples count'] = l['Samples count'].mean()
        vr, mn, n = get_variance(l['Voltage delta']);
        df_statistics.loc[row,'Voltage delta'] = l['Voltage delta'].mean();
        df_statistics.loc[row,'Voltage delta variance'] = vr;
        df_statistics.loc[row,'Mass'] = v[0]['Mass'].values[0];##############
        row += 1
    return df_statistics

def fix_bad_column_samples(df_col, window_size, sigmas_k):
    df_col_res = df_col.copy();
    variance, mean, noise = get_stationary_variance(df_col, window_size);
    print(">>> ", variance, noise.min(), noise.max());
    threshold = sigmas_k * variance;
    for n,v in enumerate(np.abs(noise)):
        if(v > threshold):
            df_col_res[n] = mean[n];
    return df_col_res;

def fix_bad_samples(ddf, window_size, sigmas_k):
    result_df = pd.DataFrame();
    result_df.index = ddf.index;
    for colName in ddf.columns:
        print(colName);
        result_df[colName] = fix_bad_column_samples(ddf.loc[:, colName], window_size, sigmas_k);
    return result_df;

def normalize_data(df_fixed):
    min_max={};
    df_normalized = pd.DataFrame()
    for n,colname in enumerate(df_fixed.columns):
        row = df_fixed[colname];
        row_min = row.min();
        row_max = row.max();
        df_normalized[colname] = (row - row_min) / (row_max - row_min);
        min_max[colname]=[row_min, row_max];
    df_normalized.index = list(range(0, df_fixed.shape[0]));
    return df_normalized, min_max

def denormalize_data(df_normalized, min_max):
    df_denormalized = pd.DataFrame(columns = df_normalized.columns,index = df_normalized.index)

    for n,colname in enumerate(df_normalized.columns):
        row_min,row_max = min_max[colname]
        scale = row_max - row_min
        df_denormalized[colname] =  df_normalized[colname] * scale + row_min
    return df_denormalized

def re_normalize_data(df_fixed, min_max):
    df_re_normalized = pd.DataFrame()
   
    for n,colname in enumerate(df_fixed.columns):
        row_min, row_max = min_max[colname];
        row = df_fixed[colname];
        df_re_normalized[colname] = (row - row_min) / (row_max - row_min);

#    df_re_normalized.index = list(range(0, df_fixed.shape[0]));  
    df_re_normalized.index = df_fixed.index;  
    return df_re_normalized, min_max
    

def copy_previous_for_nan (df):
    last_incomplete_row = -1;
    most_harmed_column = '';
    rows_count, columns_count = df.shape;
    for column in df.columns:
        for row in range(rows_count):
            if pd.isna(df.loc[row, column]):
                try:
                    previous_value = df.loc[row - 1, column];
                    df.loc[row, column] = previous_value;
                    if pd.isna(previous_value) and last_incomplete_row < row:
                        last_incomplete_row = row;
                        most_harmed_column = column;
                except (KeyError, ValueError):
                    continue;             
    return (last_incomplete_row, most_harmed_column)
    
def prepare_data_to_concatenate_new(file_path, column_names):
    df = pd.read_csv(file_path, delimiter = ",", index_col=False).loc[:, column_names];
    number, col_name = copy_previous_for_nan(df);
    df = df[number+1:]; 
    df = set_timestamp_as_index(df);
    df = df.resample('1T').mean();
    df = pad_nans(df);  
    return df

def prepare_data_to_concatenate_old(file_path, column_names):
    df = pd.read_csv(file_path, delimiter = ",", index_col=False).loc[:, column_names];
    number, col_name = copy_previous_for_nan(df);
    df = df[number+1:]; 
    df = set_timestamp_as_index_old(df);
    df = df.resample('1T').mean();
    df = pad_nans(df);  
    return df

def time_resampling_old(df, new_resampled_rate = '1T', format='%Y/%m/%d %H:%M:%S'):
    df1 = set_timestamp_as_index_old(df, format = format);
    df1 = df1.resample(new_resampled_rate).mean();
    df1 = pad_nans(df1);
    return df1
    
def time_resampling_new(df, new_resampled_rate = '1T', ts_format = '%Y/%m/%d %H:%M:%S.%f'):
    df1 = set_timestamp_as_index(df,ts_format = ts_format);
    df1 = df1.resample(new_resampled_rate).mean();
    df1 = pad_nans(df1);
    return df1
    
def collect_segments_statistics(dfs, format='%Y/%m/%d %H:%M:%S', statistics_df = pd.DataFrame()):
    dfs_result = []
    for df in dfs:
        df_tmp = get_single_segment_statistics(df, format=format)
        dfs_result.append(df_tmp)
    df = pd.concat(dfs_result); 
    df.index = list(range(0,df.shape[0]))
    return df;

def get_single_segment_statistics(df, format='%Y/%m/%d %H:%M:%S'):
    df1 = pd.DataFrame();
    df1.loc[0,'Segment'] = df['Current segment'].values[0]
    df1.loc[0,'Mass'] = df['Mass'].values[0]
    df1.loc[0,'Samples count'] = df.shape[0]

    tmp_column = pd.to_datetime(df['Timestamp'], format=format).values #'%Y-%m-%d %H:%M:%S.%f').values#format='%Y/%m/%d %H:%M:%S').values
    t1 = pd.Timestamp(tmp_column[0]);
    df1.loc[0,'Day hour'] = t1.hour #* 60 + t1.minute;
    t_min = pd.Timestamp(tmp_column.min())
    t_max = pd.Timestamp(tmp_column.max())
    df1.loc[0,'Duration'] =(t_max-t_min).seconds
    tmp_column = df['Battery cell voltage'].values
    df1.loc[0,'Start segment voltage'] = tmp_column[0]
    df1.loc[0,'End segment voltage'] = tmp_column[-1]
    df1.loc[0,'Voltage delta'] = tmp_column[-1]-tmp_column[0]
    return df1;
    