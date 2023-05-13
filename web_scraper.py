
from __future__ import print_function

import os
import sys

import numpy as np
import keras

from sentence_types import load_encoded_data
from sentence_types import encode_data, import_embedding
from sentence_types import get_custom_test_comments

from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D

# from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

from urllib.request import urlopen, urlretrieve
from bs4 import NavigableString, Tag
import bs4 as bs
import fitz
import operator

from classifier import classify
from random_question_gen import generateQuestion



def create_load_model():
    
    # Use can load a different model if desired
    model_name      = "models/2cnndense"
    embedding_name  = "data/default"
    load_model_flag = True
    arguments       = sys.argv[1:len(sys.argv)]
    if len(arguments) == 1:
        model_name = arguments[0]
        load_model_flag = os.path.isfile(model_name+".json")
    print(model_name)
    print("Load Model?", (load_model_flag))

    # Model configuration
    maxlen = 300
    batch_size = 64
    embedding_dims = 75
    pool_size = 3
    stride = 1
    filters = 100
    kernel_size = 7
    hidden_dims = 50
    epochs = 2

    # Add parts-of-speech to data
    pos_tags_flag = True

    # Export & load embeddings
    x_train, x_test, y_train, y_test = load_encoded_data(data_split=0.8,
                                                        embedding_name=embedding_name,
                                                        pos_tags=pos_tags_flag)

    word_encoding, category_encoding = import_embedding(embedding_name)

    max_words   = len(word_encoding) + 1
    num_classes = np.max(y_train) + 1

    print(max_words, 'words')
    print(num_classes, 'classes')

    print('Pad sequences (samples x time)')
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)

    print('Convert class vector to binary class matrix '
        '(for use with categorical_crossentropy)')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if not load_model_flag:

        print('Constructing model!')

        model = Sequential()
        
        model.add(Embedding(max_words, embedding_dims,
                            input_length=maxlen))
        
        model.add(Dropout(0.2))
        
        model.add(Conv1D(filters,
                        kernel_size,
                        padding='valid',
                        activation='relu',
                        strides=stride))
        
        model.add(MaxPooling1D(pool_size=pool_size))

        model.add(Dropout(0.1))

        model.add(Conv1D(filters//2,
                        kernel_size//2 + 1,
                        padding='valid',
                        activation='relu',
                        strides=1))
        
        model.add(GlobalMaxPooling1D())
        
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))    
        
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        
        model.fit(x_train, y_train, batch_size=batch_size,
                epochs=epochs, validation_data=(x_test, y_test))

        model_json = model.to_json()
        with open(model_name + ".json", "w") as json_file:
            json_file.write(model_json)
        
        # serialize weights to HDF5
        model.save_weights(model_name + ".h5")
        print("Saved model to disk")

    else:

        print('Loading model!')

        # load json and create model
        json_file = open(model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        
        # load weights into new model
        model.load_weights(model_name + ".h5")
        print("Loaded model from disk")
        
        # evaluate loaded model on test data
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])


    # score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

    # print('Test accuracy:', score[1])

    # model_name      = "models/2cnndense"
    # embedding_name  = "data/default"
    # load_model_flag = False
    # arguments       = sys.argv[1:len(sys.argv)]
    # if len(arguments) == 1:
    #     model_name = arguments[0]
    #     load_model_flag = os.path.isfile(model_name+".json")
    # print(model_name)
    # print("Load Model?", (load_model_flag))

    # # Model configuration
    # maxlen = 300
    # batch_size = 64

    # # Add parts-of-speech to data
    # pos_tags_flag = True

    # word_encoding, category_encoding = import_embedding(embedding_name)

    # max_words   = len(word_encoding) + 1
    # num_classes = np.max(y_train) + 1
    return model, model_name, embedding_name, load_model_flag, maxlen, batch_size, pos_tags_flag, max_words, num_classes

    ###############

def get_html(link_url):
    link_url=link_url.replace(" ", "%20")
    source = urlopen(link_url)
    return source

def get_soup(url):
    source = get_html(url)
    soup = bs.BeautifulSoup(source, 'lxml')
    return soup

def get_title(soup):
    title = soup.find('title')
    return title.getText()

def get_paragraphs(soup):
    """
    Use this on a BeautifulSoup's soup (use the get_soup function on a url to get it's soup) 
    in order to get the paragraphs from a url. Returns the texts in a paragraph in a list.
    
    Inputs:
    soup - Should be the "soup" of the url. Use get_soup to get the soup of the url.
    
    Outputs:
    Each paragraph's texts in a list
    """

    texts = []
    newline = ["\r\n", "\r", "\n"]
    for p in soup.findAll('p'):
        text = ""
        for child in p:
            if isinstance(child, NavigableString):
                add = str(child)
                for line in newline:
                    add = add.replace(line, " ")
                text += add
            elif isinstance(child, Tag):
                if child.name != 'br':
                    add = child.text
                    for line in newline:
                        add = add.replace(line, " ")
                    text += add
                else:
                    text += '\n'

        result = text.strip().split('\n')
        for sen in result:
            texts.append(sen)

    for p in soup.findAll('span'):
        text = ""
        for child in p:
            if isinstance(child, NavigableString):
                add = " " + str(child) + " "
                for line in newline:
                    add = add.replace(line, " ")
                text += add
            elif isinstance(child, Tag):
                if child.name != 'br':
                    add = " " + child.text + " "
                    for line in newline:
                        add = add.replace(line, " ")
                    text += add
                else:
                    text += '\n'

        result = text.strip().split('\n')
        for sen in result:
            texts.append(sen)

    for p in soup.find_all('div'):
        text = ""
        for child in p:
            if isinstance(child, NavigableString):
                add = " " + str(child) + " "
                for line in newline:
                    add = add.replace(line, " ")
                text += add
            elif isinstance(child, Tag):
                if child.name != 'br':
                    add = " " + child.text + " "
                    for line in newline:
                        add = add.replace(line, " ")
                    text += add
                else:
                    text += '\n'

        result = text.strip().split('\n')
        for sen in result:
            texts.append(sen)

    for p in soup.find_all('tr'):
        text = ""
        for child in p:
            if isinstance(child, NavigableString):
                add = " " + str(child) + " "
                for line in newline:
                    add = add.replace(line, " ")
                text += add
            elif isinstance(child, Tag):
                if child.name != 'br':
                    add = " " + child.text + " "
                    for line in newline:
                        add = add.replace(line, " ")
                    text += add
                else:
                    text += '\n'

        result = text.strip().split('\n')
        for sen in result:
            texts.append(sen)

    for p in soup.find_all('td'):
        text = ""
        for child in p:
            if isinstance(child, NavigableString):
                add = " " + str(child) + " "
                for line in newline:
                    add = add.replace(line, " ")
                text += add
            elif isinstance(child, Tag):
                if child.name != 'br':
                    add = " " + child.text + " "
                    for line in newline:
                        add = add.replace(line, " ")
                    text += add
                else:
                    text += '\n'

        result = text.strip().split('\n')
        for sen in result:
            texts.append(sen)

    return texts

def extract_sentences(texts, allow_lower_case_start = False, allow_number_start = False):
    """
    Extracts the interlingua and non-interlingua sentences from a block of text
    
    Inputs:
    texts - A list of texts that you wish to extract sentences from
    allow_lower_case_start - If this is set to True, then it will detect sentences that start with a lower case letter. If set to false, all detected sentences will start with an upper case letter. Default is False.
    allow_lower_case_start - If this is set to True, then it will detect sentences that start with a number. If set to false, all detected sentences will not start with a letter. Default is False.
 
    Outputs:
    Output - A list of all senteces the sentences in text (a combination of output 1 and 2)
    """
    
    spaces = [" ", "\u00A0"]
    newline = ["\r\n", "\r", "\n"]
    punctuation = [".", "!", "?"]
    abbreviations = ["i.a.", "i. a.", "i.e.", "i. e.", "p.ex.", "p. ex.", "sr.", "sra.", "srta.", "a.i.", "a. i.",
                     "etc.", ".html", ".com", ".org", ".net", ".int", ".edu", ".gov", ".mil",".rice", ".onet", ".se", ".pl",
                     "...", "Dr.", "Mrs.", "Mr.", "Ms.", "pp."]
    websites = ["com", "org", "net", "int", "edu", "gov", "mil","rice", "onet", "se", "pl"]
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "\"", "\'", "‘"]
    quotes = ["\"", "\'", "‘","’"]
    sentences = []
    for text in texts:
        split = text.split()
        if len(split) > 0 and split[0][-1] == ":":  # check if the first word ends with a ":"
            split.remove(split[0])

        text = " ".join(split)
        sentence_ends = [i for i, ltr in enumerate(text) if
                         ltr in punctuation]  # find period locations in sent.
        start = 0

        i = 0
        breakx = 0
        while i < len(sentence_ends):  # looking at all period locations
            interrupt = 0
            if len(sentence_ends) > 0 and sentence_ends[i] >= 3 and text[sentence_ends[i] - 3:sentence_ends[i]+1] == "www.":  # check if period is part of a link
                interrupt = 1

            if sentence_ends[i]+1<len(text) and text[sentence_ends[i]+1] not in quotes and text[sentence_ends[i]+1] != " ":
                                                                                   # if there is a character after the punctuation,
                                                                                   # it is most likely an abbreviation
                interrupt = 1

            if i + 1 < len(sentence_ends):                                                           # if there is only 1 or 2 words between punctuations
                if len(text[sentence_ends[i] + 1 : sentence_ends[i+1] + 1].split()) <= 2 or \
                                        sentence_ends[i+1] - sentence_ends[i] <= 5:                  # or less than 5 chars long
                                                                                                     # it is most likely not a sentence end.
                    interrupt = 1

            for abr in abbreviations:  # checking if the period is in an abbreviation and not a sentence end
                abr_periods = [i for i, ltr in enumerate(abr) if ltr == "."]
                for period_index in abr_periods:
                    if sentence_ends[i] - period_index >= 0 and \
                            sentence_ends[i] + len(abr) - period_index <= len(text) and \
                            text[(sentence_ends[i] - period_index):(
                                    sentence_ends[i] + len(abr) - period_index)].lower() == abr.lower():
                        interrupt = 1
                        break
                if interrupt == 1:
                    break

            if interrupt == 0:
                if text[start].isupper() or (text[start].islower() and allow_lower_case_start) or (text[start].isdigit() and allow_number_start) \
                        or (text[start] in quotes):  # check if the first letter of the sentence is a letter / quotation
                    if sentence_ends[i] + 1 < len(text) and \
                            (text[sentence_ends[i] + 1] in quotes):
                        sentences.append(text[start:sentence_ends[i] + 2])
                        sentence_start = 2
                    else:
                        sentences.append(text[start:sentence_ends[i] + 1])
                        sentence_start = 1
                else:
                    sentence_start = 1

                while True:
                    if sentence_ends[i] < len(text) - sentence_start and \
                            (text[sentence_ends[i] + sentence_start].lower() in alphabet
                    or text[sentence_ends[
                            i] + sentence_start].isdigit()):  # check if the character next to the period a space
                        start = sentence_ends[i] + sentence_start
                        break
                    elif sentence_ends[i] >= len(text) - sentence_start:
                        break
                    else:
                        sentence_start = sentence_start + 1

            i = i + 1
    i = 0
    while i < len(sentences):
        if len(sentences[i].split()) <= 2 and len(sentences[i]) <= 5:  # If a "sentence" has 2 or less words and less
                                                                    #    than or equal to 5 chars, delete it
            sentences.remove(sentences[i])
            i = i - 1
        i = i + 1

    return sentences

def get_pdf_text(url, is_url=True):
    """
    Used to get the paragraph text out of a pdf.
    
    Inputs:
    url - The url of the pdf or the name of the pdf in the same directory
    is_url - If the pdf is from a url, this should be True. If it's a pdf in your directory, this should be False. Deafult is True.
    
    Output:
    A list of each paragraph in the pdf.
    """
    
    out = []
    if is_url:
        urlretrieve(url, 'temp.pdf')
        doc = fitz.open("temp.pdf")
    else:
        doc = fitz.open(url)

    font = fonts(doc)

    tags = font_tags(font[0], font[1])

    headers = headers_para(doc, tags)

    for text in headers:
        if(text.find("<p>")==0):
            out.append(" ".join(text[3:].replace("|"," ").split()))

    return out

def text_from_txt(url, is_url=True):
    """
    Used to get the text out of a txt file.
    
    Inputs:
    url - The url of the txt file or the name of the txt file in the same directory
    is_url - If the txt file is from a url, this should be True. If it's a txt file in your directory, this should be False. Deafult is True.
    
    Output:
    A list of each paragraph in the txt file.
    """
    
    continuation = ["-\r\n","-\r","-\n"]
    linebreaks = ["\r\n","\r","\n"]
    if is_url:
        source = urlopen(url)

        lines = []
        text = ""

        for line in source:
            line = line.decode("utf-8")
            if(line in linebreaks):
                if text != "":
                    lines.append(text)
                text = ""
            else:
                for cont in continuation:
                    line = line.replace(cont, "")

                for breaks in linebreaks:
                    line = line.replace(breaks, " ")
                text += line
        lines.append(text)
        return lines
    else:
        file = open(url, "r")
        file_lines = file.readlines()
        text = ""
        lines = []

        for line in file_lines:
            if(line in linebreaks):
                if text != "":
                    lines.append(text)
                text = ""
            else:
                for cont in continuation:
                    line = line.replace(cont, "")
                for breaks in linebreaks:
                    line = line.replace(breaks, " ")
                text += line
        lines.append(text)
        return lines

def fonts(doc, granularity=False):  # thanks to https://towardsdatascience.com/extracting-headers-and-paragraphs-from-pdf-using-pymupdf-676e8421c467
    """Extracts fonts and their usage in PDF documents.
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param granularity: also use 'font', 'flags' and 'color' to discriminate text
    :type granularity: bool
    :rtype: [(font_size, count), (font_size, count}], dict
    :return: most used fonts sorted by count, font style information
    """
    styles = {}
    font_counts = {}

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # block contains text
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if granularity:
                            identifier = "{0}_{1}_{2}_{3}".format(s['size'], s['flags'], s['font'], s['color'])
                            styles[identifier] = {'size': s['size'], 'flags': s['flags'], 'font': s['font'],
                                                  'color': s['color']}
                        else:
                            identifier = "{0}".format(s['size'])
                            styles[identifier] = {'size': s['size'], 'font': s['font']}

                        font_counts[identifier] = font_counts.get(identifier, 0) + 1  # count the fonts usage

    font_counts = sorted(font_counts.items(), key=operator.itemgetter(1), reverse=True)

    if len(font_counts) < 1:
        raise ValueError("Zero discriminating fonts found!")

    return font_counts, styles

def font_tags(font_counts, styles):  # thanks to https://towardsdatascience.com/extracting-headers-and-paragraphs-from-pdf-using-pymupdf-676e8421c467
    """Returns dictionary with font sizes as keys and tags as value.
    :param font_counts: (font_size, count) for all fonts occuring in document
    :type font_counts: list
    :param styles: all styles found in the document
    :type styles: dict
    :rtype: dict
    :return: all element tags based on font-sizes
    """
    p_style = styles[font_counts[0][0]]  # get style for most used font by count (paragraph)
    p_size = p_style['size']  # get the paragraph's size

    # sorting the font sizes high to low, so that we can append the right integer to each tag
    font_sizes = []
    for (font_size, count) in font_counts:
        font_sizes.append(float(font_size))
    font_sizes.sort(reverse=True)

    # aggregating the tags for each font size
    idx = 0
    size_tag = {}
    for size in font_sizes:
        idx += 1
        if size == p_size:
            idx = 0
            size_tag[size] = '<p>'
        if size > p_size:
            size_tag[size] = '<h{0}>'.format(idx)
        elif size < p_size:
            size_tag[size] = '<s{0}>'.format(idx)

    return size_tag


def headers_para(doc, size_tag):  # thanks to https://towardsdatascience.com/extracting-headers-and-paragraphs-from-pdf-using-pymupdf-676e8421c467
    """Scrapes headers & paragraphs from PDF and return texts with element tags.
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param size_tag: textual element tags for each size
    :type size_tag: dict
    :rtype: list
    :return: texts with pre-prended element tags
    """
    header_para = []  # list with headers and paragraphs
    first = True  # boolean operator for first header
    previous_s = {}  # previous span

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # this block contains text

                # REMEMBER: multiple fonts and sizes are possible IN one block

                block_string = ""  # text found in block
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if s['text'].strip():  # removing whitespaces:
                            if first:
                                previous_s = s
                                first = False
                                block_string = size_tag[s['size']] + s['text']
                            else:
                                if s['size'] == previous_s['size']:

                                    if block_string and all((c == "|") for c in block_string):
                                        # block_string only contains pipes
                                        block_string = size_tag[s['size']] + s['text']
                                    if block_string == "":
                                        # new block has started, so append size tag
                                        block_string = size_tag[s['size']] + s['text']
                                    else:  # in the same block, so concatenate strings
                                        block_string += " " + s['text']

                                else:
                                    header_para.append(block_string)
                                    block_string = size_tag[s['size']] + s['text']

                                previous_s = s

                    # new block started, indicating with a pipe
                    block_string += "|"

                header_para.append(block_string)

    return header_para



def scrape(urls, id):
    # urls = ["https://people.eecs.berkeley.edu/~jrs/189/lec/05.pdf","https://people.eecs.berkeley.edu/~jrs/189/lec/06.pdf","https://people.eecs.berkeley.edu/~jrs/189/hw/hw3.pdf","https://people.eecs.berkeley.edu/~jrs/189/lec/02.pdf","https://people.eecs.berkeley.edu/~jrs/189/lec/01.pdf","https://people.eecs.berkeley.edu/~jrs/189/lec/03.pdf","https://people.eecs.berkeley.edu/~jrs/189/lec/04.pdf","https://people.eecs.berkeley.edu/~jrs/189/"]

    for url in urls:
        try:
            if url.find(".pdf")!=-1:
                print("This is a pdf")
                url_tag = "pdf"
                sentences = extract_sentences(get_pdf_text(url), allow_number_start=False)
            elif url.find(".txt")!=-1:
                print("This is a text file")
                url_tag = "text"
                lines = text_from_txt(url)
                sentences = extract_sentences(lines)
            else:
                print("This is a website: " + url)
                url_tag = "website"
                soup = get_soup(url)
                paragraphs = get_paragraphs(soup)
                sentences = extract_sentences(paragraphs)

        except:
            pass
    return sentences
   

def classify_helper(sentences, keywords, model, model_name, embedding_name, load_model_flag, maxlen, batch_size, pos_tags_flag, max_words, num_classes):
    #create keyword indices
    keyword_idxs = {key: [] for key in keywords}
    sentences = np.array(sentences)
    all_sentences = []
    to_classify = []
    for sentence in sentences:
        all_sentences.append(sentence + "\n")
        # to_classify.append(str(sentence).lower())
        for keyword in keywords:
            if (keyword in str(sentence).lower()):
                to_classify.append(str(sentence).lower())
                # keyword_idxs[keyword].append(i)
                break

    classifications = classify(to_classify, model, model_name, embedding_name, load_model_flag, maxlen, batch_size, pos_tags_flag, max_words, num_classes)
    classifications = np.argmax(classifications, axis=1) % 2
    idxs = np.where(classifications == 1)[0]
    print(idxs)
    print(sentences[idxs])
    result_sentences = sentences[idxs]

    f = open(f"./{id}_output.txt", "w")
    f.writelines(sentences)
    f.close()

    f = open(f"./{id}_output_questions.txt", "w")
    f.writelines(result_sentences)
    f.close()



def main():
    #urls = ["https://people.eecs.berkeley.edu/~jrs/189/lec/05.pdf","https://people.eecs.berkeley.edu/~jrs/189/lec/06.pdf","https://people.eecs.berkeley.edu/~jrs/189/hw/hw3.pdf","https://people.eecs.berkeley.edu/~jrs/189/lec/02.pdf","https://people.eecs.berkeley.edu/~jrs/189/lec/01.pdf","https://people.eecs.berkeley.edu/~jrs/189/lec/03.pdf","https://people.eecs.berkeley.edu/~jrs/189/lec/04.pdf","https://people.eecs.berkeley.edu/~jrs/189/"]
    urls = ["https://openstax.org/books/algebra-and-trigonometry/pages/1-1-real-numbers-algebra-essentials"]
    # urls = ["https://yoshiwarabooks.org/mfg/mfg.pdf"]

    #desired keywords may differ between textbooks. Try to read parts of textbook to see
    keywords = ["write", "simplify", "please", "show", "give"]

    model, model_name, embedding_name, load_model_flag, maxlen, batch_size, pos_tags_flag, max_words, num_classes = create_load_model()
    sentences = scrape(urls, 5)
    classify_helper(sentences, keywords, model, model_name, embedding_name, load_model_flag, maxlen, batch_size, pos_tags_flag, max_words, num_classes)


    sample_question = "What is 5 * 7?"
    fileNames = ["5_output_sentences.txt"] #TF-IDF on these
    generateQuestion(sample_question, fileNames)
if __name__ == "__main__":
    main()