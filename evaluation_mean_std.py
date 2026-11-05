import os
import pandas as pd

main_dir = '/mnt/disk1/Code_x/HGT-main/parameter_ResNet_repeat/40subject_Nb0_sh_w1_1_w2_1_w3_0001/lr0.001/test_model_99/'

excel_filename = 'test_epoch200_lr0.001metrics.xlsx'

output_path = '/mnt/disk1/Code_x/HGT-main/parameter_ResNet_repeat/40subject_Nb0_sh_w1_1_w2_1_w3_0001/lr0.001/test_model_99/test.xlsx'

all_data = []

for subdir in os.listdir(main_dir):
    subdir_path = os.path.join(main_dir, subdir)
    if os.path.isdir(subdir_path):
        excel_path = os.path.join(subdir_path, excel_filename)
        if os.path.exists(excel_path):

            df = pd.read_excel(excel_path, index_col=0)
            all_data.append(df)

if all_data:

    combined_df = pd.concat(all_data, axis=0)

    mean_df = combined_df.groupby(combined_df.index).mean()
    std_df = combined_df.groupby(combined_df.index).std()

    result_df = pd.concat([mean_df, std_df], axis=0, keys=['Mean', 'Std'])

    index = result_df.index

    mean_mean_idx = index.get_loc(('Mean', 'mean'))  
    mean_od_idx = index.get_loc(('Mean', 'od'))  
    std_mean_idx = index.get_loc(('Std', 'mean'))  
    std_od_idx = index.get_loc(('Std', 'od'))  

    result_df.iloc[[mean_mean_idx, mean_od_idx]] = result_df.iloc[[mean_od_idx, mean_mean_idx]].values

    result_df.iloc[[std_mean_idx, std_od_idx]] = result_df.iloc[[std_od_idx, std_mean_idx]].values

    new_index = list(index)
    new_index[mean_mean_idx], new_index[mean_od_idx] = new_index[mean_od_idx], new_index[mean_mean_idx]

    new_index[std_mean_idx], new_index[std_od_idx] = new_index[std_od_idx], new_index[std_mean_idx]

    result_df.index = pd.MultiIndex.from_tuples(new_index)
    # print(result_df.index.values[[6]])
    # print(result_df.iloc[[6], 1])

    result_df.to_excel(output_path)

    print(f"The result has been saved to: {output_path}")
else:
    print("No Excel files found.")

