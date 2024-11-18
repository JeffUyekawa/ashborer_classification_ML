
#%%
import pandas as pd
def prepare_multiclass(in_path, out_path):
    df = pd.read_csv(in_path)
    df_0 = df[df['Label'] == 0]
    grouped = df_0.groupby(by=['File', 'Start'])
    for i, group in grouped:
        if group['Roll Amount'].mean() > 0:
            file = group['File'].iloc[0]
            start = group['Start'].iloc[0]
            df.loc[(df['File'] == file) & (df['Start'] == start), 'Label'] = 2
    df.to_csv(out_path, index = False)

train_df = pd.read_csv(r"C:\Users\jeffu\Documents\Recordings\new_training_data.csv")
test_df = pd.read_csv(r"C:\Users\jeffu\Documents\Recordings\new_test_data.csv")

# %%
in_path = r"C:\Users\jeffu\Documents\Recordings\new_test_data.csv"
out_path = r"C:\Users\jeffu\Documents\Recordings\multiclass_test_data.csv"
prepare_multiclass(in_path,out_path)
# %%
