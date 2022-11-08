import pickle
from google_play_scraper import Sort, reviews_all
from multiprocessing import Pool


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


def get_reviews(pkg_name):
    try:
        result = reviews_all(
                        pkg_name,
                        sleep_milliseconds=0, # defaults to 0
                        lang='en', # defaults to 'en'
                        country='us', # defaults to 'us'
                        sort=Sort.MOST_RELEVANT, # defaults to Sort.MOST_RELEVANT
                        # filter_score_with=5 # defaults to None(means all score)
        )
    except:
        result = 'App not found'
    return result


def main_single(pkg_name_list):
    for pkg_name in pkg_name_list:
        result = pkg_name + '->'+str(get_reviews(pkg_name))
        write_result(result, 'model_app_reviews.txt')


def get_apk_double_list(google_play_apk_list):
    apk_double_list = []
    left = 0
    while left <= len(google_play_apk_list):
        tmp_list = google_play_apk_list[left: left+915]
        apk_double_list.append(tmp_list)
        left += 915
    return apk_double_list


google_play_apk_list = pickle.load(open('google_play_apk_list.pkl', 'rb'))
apk_double_list = get_apk_double_list(google_play_apk_list)

with Pool(28) as p:
    p.map(main_single, apk_double_list)

