import os
import re
from string import punctuation
from nltk import PorterStemmer, word_tokenize
# import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords
from dependencies import lovinss as lovins


def load_corpus(lower_case=False, remove_punctuation=False, remove_stopwors=False, remove_urls=False,
                remove_mentions=False, use_stemmer=False, pre_filename=None):
    datasets = []
    source_dir = os.path.join(os.getcwd(), "data", "labeled")

    all_files = os.listdir(source_dir)
    stem_err_counter = 0
    for i, filename in enumerate(all_files):
        file_path = os.path.join(source_dir, filename)
        if os.path.isfile(file_path):
            if pre_filename != None and not filename.lower().startswith(pre_filename):
                continue
            dataset = []
            with open(file_path, encoding="UTF-8", errors='ignore') as file_reader:
                first_line = file_reader.readline()
                for line in file_reader:
                    line_parts = line.split("\t")
                    label = line_parts[-1].replace("\n", "")

                    id = int(line_parts[0].replace("\'", ""))
                    post = line_parts[3]

                    clean_post = post.strip()

                    if remove_mentions:
                        clean_post = re.sub(r"[@][\w_-]+", '', clean_post)

                    if remove_urls:
                        clean_post = re.sub(r'http\S+', '', clean_post)

                    if lower_case:
                        clean_post = clean_post.lower()

                    if remove_punctuation:
                        translation = str.maketrans("", "", punctuation)
                        clean_post = clean_post.translate(translation)

                    if remove_stopwors:
                        stop_words = set(stopwords.words('english'))
                        clean_post = ' '.join([word for word in clean_post.split() if word not in stop_words])

                    if use_stemmer:
                        ps = PorterStemmer()
                        words = word_tokenize(clean_post)
                        stems = []
                        for w in words:
                            try:
                                stem = lovins.stem(w)
                            except:
                                stem_err_counter = stem_err_counter + 1
                                print(stem_err_counter)
                                stem = ps.stem(w)
                                # print("An exception occurred")

                            stems.append(stem)
                            # print(stem)

                        clean_post = ' '.join(stems)

                    dataset.append({"id": id, "label": label, "text": clean_post, "file": filename})

            datasets.append({filename: dataset})

    return datasets