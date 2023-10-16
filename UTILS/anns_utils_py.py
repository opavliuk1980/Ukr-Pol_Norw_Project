import pandas as pd
import numpy as np
#from data_reformat import df_to_shifted_tables
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

def predictions_to_df(predictions, columns):
    if len(columns) == 1: return pd.DataFrame(predictions, columns = columns);
    predictions_df = pd.DataFrame()
    for n, p in enumerate(predictions):
        predictions_df[columns[n]] = pd.DataFrame(p);
    return predictions_df;

def predict_time_window(df_fixed, model, PREDICTIONS_COUNT):
    inputs_count = model.inputs[0].shape[1];
    zeroes = [0. for x in range(len(model.inputs))];
    df_f = df_fixed[-inputs_count:];
    df_0 = pd.DataFrame([zeroes], columns = df_f.columns)
    df_predictions = pd.DataFrame(columns = df_fixed.columns);
    
    for t in range(PREDICTIONS_COUNT):
        df_extended = pd.concat([df_f, df_0]);
        x_train, y_train, x_test, y_test = df_to_shifted_tables(df_extended, inputs_count, 0);
        predictions = model.predict(x_test);
        #df_predicted = pd.DataFrame([[p[0][0] for p in predictions]], columns = df_f.columns);
        row_data = list(np.array(predictions).flat);
        df_predicted = pd.DataFrame([row_data], columns = df_fixed.columns);
        df_f = pd.concat([df_f, df_predicted])[1:];
        df_predictions = pd.concat([df_predictions, df_predicted]);

    next_index = df_fixed.index[-1] + 1;
    df_predictions.index = list(range(next_index, next_index + PREDICTIONS_COUNT));
    return df_predictions

def predict_time_window_with_knowns(df_fixed, model, outputs_names, df_knowns):
    inputs_count = model.inputs[0].shape[1];
    zeroes = [0. for x in range(len(model.inputs))];
    print(df_fixed.shape)
    df_f = df_fixed[-inputs_count:];
    df_0 = pd.DataFrame([zeroes], columns = df_f.columns)
    df_predictions = pd.DataFrame(columns = df_fixed.columns);
    
    for t in df_knowns.index:
        df_extended = pd.concat([df_f, df_0]);
        x_train, y_train, x_test, y_test = df_to_shifted_tables(df_extended, inputs_count, 0);
        
        predictions = model.predict(x_test);
        row_data = list(np.array(predictions).flat);

        df_predicted = pd.DataFrame([row_data], columns = outputs_names);
        df_predicted.loc[0, df_knowns.columns] = df_knowns.loc[t,:]
        df_f = pd.concat([df_f, df_predicted])[1:];
        df_predictions = pd.concat([df_predictions, df_predicted]);
    df_predictions.index = df_knowns.index;
    return df_predictions[outputs_names]

def df_to_shifted_tables(source_df, max_shift, count_trein=100):
    input_trein_dfs = []
    output_trein_dfs = []
    input_test_dfs = []
    output_test_dfs = []
    for col in source_df.columns:
        print(" -- " + col)
        input_df, output_df = column_to_shifted_table(source_df[[col]], max_shift)
        input_trein_dfs.append({col:input_df[:count_trein]})
        output_trein_dfs.append(output_df.iloc[:count_trein])
        input_test_dfs.append({col:input_df[count_trein:]})
        output_test_dfs.append(output_df.iloc[count_trein:])
    return (input_trein_dfs, pd.DataFrame(output_trein_dfs).transpose(), input_test_dfs, pd.DataFrame(output_test_dfs).transpose())

def column_to_shifted_table(df_column, max_shift):
    column_name = df_column.columns[0];
    shifted_data = pd.DataFrame();
    for i in range (max_shift, 0, -1):
        shifted_data['t-' + str(i)] = df_column.iloc[:,0].shift(i)
    shifted_data[column_name] = df_column.iloc[:,0].values
    trunc_data = shifted_data[max_shift:]
    outputs = trunc_data[column_name]
    inputs = trunc_data.drop(column_name, axis = 1)
    return (inputs, outputs)

def build_multiparamiters_model(input_parameters_names, output_parameters_names, inputs_per_parameters = 12, hiden_layer_neurons = 8):
    inputs = [];
    for col in input_parameters_names:
        inp = Input(shape=(inputs_per_parameters), name="Inp_" + col.replace(" ", "_"));
        inputs.append(inp);
    
    outputs = []
    merged_inputs = layers.concatenate(inputs);
    for col in output_parameters_names:
        col_label = col.replace(" ", "_");
        out = Dense(hiden_layer_neurons, activation='relu', name="mid_" + col_label)(merged_inputs);
        out = Dense(1, activation='linear', name="out_" + col_label)(out);
        outputs.append(out);

    model = Model(inputs=inputs, outputs=outputs);
#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'],run_eagerly=True)

    return model

def build_multiparamiters_model_single_output(parameters_names, inputs_per_parameters = 12, hiden_layer_neurons = 8):
    inputs = [];
    for col in parameters_names:
        inp = Input(shape=(inputs_per_parameters), name="Inp_" + col.replace(" ", "_"));
        inputs.append(inp);
    
    outputs = []
    merged_inputs = layers.concatenate(inputs);
    col= 'result';
    col_label = col.replace(" ", "_");
    out = Dense(hiden_layer_neurons, activation='relu', name="mid_" + col_label)(merged_inputs);
    out = Dense(1, activation='linear', name="out_" + col_label)(out);
    outputs.append(out);

    model = Model(inputs=inputs, outputs=outputs);
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
    return model