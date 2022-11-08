import pandas as pd
from transformers import pipeline

key_words_list = ['prediction', 'predict', 'predicted', 'predictor', 'predicting',
                  'forecast', 'forecasts', 'forecasting', 'forecasted',
                  'machine learning', 'deep learning', 'classify', 'classification',
                  'accuracy', 'accurate', 'performance', 'performances'
                  ]


def get_acc_reviews(df):
    res_reviews = []
    res_score = []
    content_list = list(df['content'])
    score_list = list(df['score'])
    for i in range(len(content_list)):
        for key in key_words_list:
            if key in str(content_list[i]):
                res_reviews.append(content_list[i])
                res_score.append(score_list[i])
                continue
    df = pd.DataFrame(columns=['score', 'content'])
    df['score'] = res_score
    df['content'] = res_reviews
    return df


def get_positive_negative(df):
    classifier = pipeline('sentiment-analysis')
    label_list = []

    for review in df['content']:
        try:
            res = classifier(review)
            res = res[0]['label']
            label_list.append(res)
        except:
            print(review)
            label_list.append('Wrong')
    df['label'] = label_list
    df = df[df['label'] != 'Wrong']
    return df


def get_score_analysis(df):
    for i in range(1, 6):
        tmp_df = df[df['score']==i]
        n_p = len(tmp_df[tmp_df['label']=='POSITIVE'])
        n_n = len(tmp_df[tmp_df['label']=='NEGATIVE'])
        print('score=',i)
        print('POSITIVE=', n_p)
        print('NEGATIVE=', n_n)
        print('=========')

df = pd.read_csv('reviews.csv')
print(len(df))
df = get_acc_reviews(df)
print(len(df))
df = get_positive_negative(df)
print(len(df))

df.to_csv("tmp_label.csv", index=False, sep=',')
print('positive:', len(df[df['label']=='POSITIVE']))
print('negative:', len(df[df['label']=='NEGATIVE']))

df = pd.read_csv('tmp_label.csv')
get_score_analysis(df)





