import pandas as pd
import numpy as np
from scipy import stats

if __name__ == "__main__":
    all_csv_files = ['b_lstm+(1).csv', 'b_lstm.csv', 'cb_.csv', 'cb_ (1).csv', 'cb_ (2).csv', 'cb_cv_tsfr.csv', 'cb_tsfr.csv']
    df_list = [pd.read_csv(file) for file in all_csv_files]

    new_df_list = []
    new_df_list.extend([df_list[0], df_list[1], df_list[2], df_list[-1], df_list[-2]])
    df_list = new_df_list

    df_list_list = []
    for item in df_list:
        df_list_list.append(item["label"].values)

    df_list_list = np.array(df_list_list)

    # Compute mode value using scipy
    mode_value_scipy = stats.mode(df_list_list, axis=0)[0][0]

    assert (mode_value_scipy.shape[0] == 450)

    new_df = pd.DataFrame({"label": mode_value_scipy})
    new_df.to_csv("blender.csv", index=False)