# OAT
This project is an OATutor ML spi-off seeking to improve the content team's velocity of question generation by solving the following:
1: automatic parsing out of questions from pdf or web-based creative commons textbooks
Run web_scraper.py as the "main" function
Contains a web scraper that parses a PDF/website into a string file
Then, to determine whether a sentence is a "homework" question/problem, uses sentence type classifier. If a sentence is a command or question, then it is probably a homework question. 
-Credits for the sentence type classifier go to Austin Walters (https://austingwalters.com)



2: automatic permutation of the questions using ML (e.g., LLM)
My code identifies what terms are exchangeable in a question for other phrases. 
The technique I use is TF-IDF, where a terms frequency in a text will determine it's relevance. I build a TF-IDF index on the input document (a page of the textbook), and then use it to identify and replace words in a different sample question. The terms will be printed out. This code is in random_question_gen, called from web_scraper.py main function.
