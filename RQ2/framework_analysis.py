import pandas as pd


def udf_tflite(s):
    if 'TfLite' in s:
        return 1
    else:
        return 0


def udf_caffe(s):
    if 'Caffe' in s:
        return 1
    else:
        return 0


def udf_mallet(s):
    if 'MALLET' in s:
        return 1
    else:
        return 0


def udf_neuroph(s):
    if 'Neuroph' in s:
        return 1
    else:
        return 0


df_tmp_label = pd.read_csv('/reviews_analysis/tmp_label.csv')
df_reviews = pd.read_csv('/reviews_analysis/reviews.csv')
df_all = pd.read_csv('/reviews_analysis/df_all.csv')
df_merge = df_tmp_label.merge(df_reviews, left_on='content', right_on='content', how='inner')
df_merge = df_merge.merge(df_all, left_on='apk_name', right_on='pkg_name', how='left')

df_merge['tflite'] = df_merge['framework_list'].apply(udf_tflite)
df_merge['caffe'] = df_merge['framework_list'].apply(udf_caffe)
df_merge['mallet'] = df_merge['framework_list'].apply(udf_mallet)
df_merge['neuroph'] = df_merge['framework_list'].apply(udf_neuroph)


# tflite
df_tflite = df_merge[df_merge['tflite']==1]
df_caffe = df_merge[df_merge['caffe']==1]
df_mallet = df_merge[df_merge['mallet']==1]
df_neuroph = df_merge[df_merge['neuroph']==1]
print('tflite:', 'POSITIVE', len(df_tflite[df_tflite['label']=='POSITIVE']), 'NEGATIVE', len(df_tflite[df_tflite['label']=='NEGATIVE']))
print('caffe:', 'POSITIVE', len(df_caffe[df_caffe['label']=='POSITIVE']), 'NEGATIVE', len(df_caffe[df_caffe['label']=='NEGATIVE']))
print('mallet:', 'POSITIVE', len(df_mallet[df_mallet['label']=='POSITIVE']), 'NEGATIVE', len(df_mallet[df_mallet['label']=='NEGATIVE']))
print('neuroph:', 'POSITIVE', len(df_neuroph[df_neuroph['label']=='POSITIVE']), 'NEGATIVE', len(df_neuroph[df_neuroph['label']=='NEGATIVE']))

