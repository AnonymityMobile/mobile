import pandas as pd


def get_reviews_csv(path):
    f = open(path, 'r')
    lines = f.readlines()
    lines_list = [i.strip() for i in lines if len(i) > 10]
    apk_name = [i.split('->')[0].strip() for i in lines_list]
    inf_list = [i.split('->')[1].strip() for i in lines_list]
    new_apk_name_list = []
    new_inf_list = []
    for i in range(len(inf_list)):
        if inf_list[i] != '[]':
            new_inf_list.append(inf_list[i])
            new_apk_name_list.append(apk_name[i])

    content_list = []
    score_list = []
    apk_list = []

    for i in range(len(new_apk_name_list)):
        tmp_inf_list = new_inf_list[i]
        tmp_inf_list =tmp_inf_list.replace('datetime.datetime', '')
        try:
            tmp_inf_list = eval(tmp_inf_list)
            for dic in tmp_inf_list:
                content_list.append(dic['content'])
                score_list.append(dic['score'])
                apk_list.append(new_apk_name_list[i])

        except:
            pass

    df = pd.DataFrame(columns=['apk_name', 'score', 'content'])
    df['apk_name'] = apk_list
    df['score'] = score_list
    df['content'] = content_list
    df.to_csv("reviews.csv", index=False, sep=',')
    print(len(df))

get_reviews_csv('model_app_reviews.txt')








