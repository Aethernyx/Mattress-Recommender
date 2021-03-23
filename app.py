from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, template_folder='./templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    for rendering results on HTML GUI
    '''
    features = [str(x) for x in request.form.values()]
    print(features)
    final_features = 'I am a {}, and I prefer a {} {} mattress around {} dollars. {}'.format(features[3], features[1],
                                                                                          features[2], features[0],
                                                                                             features[-1])

    best_mattress = mattress_recommender(final_features)

    return render_template('index.html', prediction_text=best_mattress)



def mattress_recommender(text):
    original_data = 'df_w_key_words.csv'
    df = pd.read_csv(original_data, lineterminator='\n')
    df.drop('Unnamed: 0', axis=1, inplace=True)
        
    r = Rake()
    tfidf = TfidfVectorizer()
    r.extract_keywords_from_text(text)
    test_key_words_dict_score = r.get_word_degrees()
    cleaned_text = [' '.join(list(test_key_words_dict_score.keys()))]

    production_df = df.copy()

    price_types = ['<500', '500-1000', '1001-1500', '1501-2000', '2001-2500', '2500+']
    bed_types = ['Foam', 'Hybrid']

    for bed_size in df.bedsize.unique():
        if bed_size.lower() in cleaned_text[0].lower():
            production_df = production_df[production_df['bedsize'] == bed_size].reset_index(drop=True)

    for price in price_types:
        if price in text:
            if '-' in text:
                num_1 = int(text.split('-')[0][-4:].replace(' ', ''))
                num_2 = int(text.split('-')[1][:4])
                production_df = production_df[(production_df['price'] >= num_1) & \
                                              (production_df['price'] <= num_2)].reset_index(drop=True)
            elif '<' in text:
                number = int(text.split('<')[1][:3])
                production_df = production_df[production_df['price'] <= number].reset_index(drop=True)
            elif '+' in text:
                number = int(text.split('+')[1][:4])
                production_df = production_df[production_df['price'] >= number].reset_index(drop=True)

    for bed_type in bed_types:
        if bed_type.lower() in cleaned_text[0].lower():
            production_df = production_df[production_df['bedtype'] == bed_type].reset_index(drop=True)

    if len(production_df) < 1:
        production_df = df.copy()
        for bed_size in df.bedsize.unique():
            if bed_size.lower() in cleaned_text[0].lower():
                production_df = production_df[production_df['bedsize'] == bed_size].reset_index(drop=True)
    else:
        pass

    tfidf_matrix = tfidf.fit_transform(production_df['key_words'])
    text_matrix = tfidf.transform(cleaned_text)

    results = pd.DataFrame(cosine_similarity(tfidf_matrix, text_matrix))

    mattress_result_count = production_df.copy()
    mattress_result_count = mattress_result_count[0:0]

    for index, row in results.sort_values(0, ascending=False).head(50).iterrows():
        mattress_result_count = mattress_result_count.append(production_df.loc[index, ['brand',
                                                                                       'model', 'bedsize',
                                                                                       'price']])
        mattress_result_count.loc[index, 'similarity_score'] = row[0]

    most_similar = mattress_result_count.groupby(
        ['brand', 'model', 'price', 'bedsize']).count().similarity_score.reset_index(). \
        sort_values(['similarity_score'], ascending=False).reset_index(drop=True)

    top_mattress = most_similar.iloc[0]

    return '''The best Mattress for you is a: \n
    {} {} {}, which is ${}'''.format(top_mattress['bedsize'],
                                     top_mattress['brand'].title(), top_mattress['model'],
                                     int(top_mattress['price']))



'''
@app.route('/predict_api', methods=['POST'])
def predict_api():
    
    #for direct API calls
    
    data = request.get_json(force=True)



    best_mattress = mattress_recommender(final_features)


    output =
    return jsonify(output)
    '''

if __name__ == '__main__':
    app.run(debug=True)
