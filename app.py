import csv
import re
import itertools
import string
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow_text as text

import tensorflow as tf
from urllib.request import urlopen as uReq

import requests
from bs4 import BeautifulSoup as bs
from flask import Flask, render_template, request
from flask_cors import cross_origin

app = Flask(__name__)


savedModel = tf.keras.models.load_model(
    ('gfgModel.h5'), compile=False,
    custom_objects={'KerasLayer': hub.KerasLayer}
)


def list_to_csv(data_list, file_path):
    fieldnames = ["Product", "Name", "Rating", "CommentHead", "Comment"]

    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_list)


def sentiment(k):
    cnt = 0
    val = 0
    for x in k:
        x = re.sub(r'[^\w\s]', '', x)  # Remove punctuation
        x = x.encode('ascii', 'ignore').decode('ascii')
        cnt += 1
        val += np.array(savedModel.predict([x])).argmax()
    if (cnt == 0):
        return "No reviews Found"
    sen = round(val / cnt)
    if (sen == 0):
        return "Negative"
    if (sen == 1):
        return "Neutral"
    if (sen == 2):
        return "Positive"
    if (sen == 3):
        return "Highly Positive"


@app.route('/', methods=['GET'])  # route to display the home page
# @cross_origin()
def homePage():
    return render_template("index.html")


# route to show the review comments in a web UI
@app.route('/review', methods=['POST', 'GET'])
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            searchString = request.form['content'].replace(" ", "")
            flipkart_url = "https://www.flipkart.com/search?q=" + searchString
            uClient = uReq(flipkart_url)
            flipkartPage = uClient.read()
            uClient.close()
            flipkart_html = bs(flipkartPage, "html.parser")
            bigboxes = flipkart_html.findAll(
                "div", {"class": "_1AtVbE col-12-12"})
            del bigboxes[0:3]
            box = bigboxes[0]
            productLink = "https://www.flipkart.com" + \
                box.div.div.div.a['href']
            prodRes = requests.get(productLink)
            prodRes.encoding = 'utf-8'
            prod_html = bs(prodRes.text, "html.parser")
            print(prod_html)
            commentboxes = prod_html.find_all('div', {'class': "_16PBlm"})

            filename = searchString + ".csv"
            fw = open(filename, "w")
            headers = "Product, Customer Name, Rating, Heading, Comment \n"
            fw.write(headers)
            reviews = []
            for commentbox in commentboxes:
                try:

                    name = commentbox.div.div.find_all(
                        'p', {'class': '_2sc7ZR _2V5EHH'})[0].text
                    name.encode(encoding='utf-8')
                except:
                    name = 'No Name'

                try:

                    rating = commentbox.div.div.div.div.text
                    rating.encode(encoding='utf-8')

                except:
                    rating = 'No Rating'

                try:

                    commentHead = commentbox.div.div.div.p.text
                    commentHead.encode(encoding='utf-8')
                except:
                    commentHead = 'No Comment Heading'
                try:
                    comtag = commentbox.div.div.find_all('div', {'class': ''})
                    custComment = comtag[0].div.text
                    custComment.encode(encoding='utf-8')
                except Exception as e:
                    print("Exception while creating dictionary: ", e)

                mydict = {"Product": searchString, "Name": name, "Rating": rating, "CommentHead": commentHead,
                          "Comment": custComment}
                reviews.append(mydict)
            sentimentList = []
            for x in reviews:
                sentimentList.append(x['Comment'])
            print(sentimentList)
            list_to_csv(reviews, filename)
            reviewSentiment = sentiment(sentimentList)
            return render_template('results.html', reviews=reviews[0:(len(reviews) - 1)], reviewSentiment=reviewSentiment)
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True)
