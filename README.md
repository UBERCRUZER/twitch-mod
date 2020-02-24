# twitch-mod
Twitch neural net moderator based on gensim doc2vec encoding and Keras

1. use doc2vec_train,py to create parameters to encode corpus.

2. use moderator.py to train moderator neural net and detect offensive messages. 


This project is loosely based off of prior work on twitter hate speech detection using doc2vec encoding. See: https://beta.vu.nl/nl/Images/werkstuk-biere_tcm235-893877.pdf

Chat lines were logged from partnered twitch streamer, martinimonsters (https://www.twitch.tv/martinimonsters/) using an IRC logger over a one month period.

I was only able to gather about 10k "useful" lines of chat after whittling down about 100k lines from unhelpful (in regards to training this model) one line responses and emote spam eg. "k" or "Kappa Kappak Kappa Kappa". While this wasn't enough to get useful results, I do still believe the concept is solid due to its effectiveness on kaggle's twitter hate speech dataset (https://www.kaggle.com/vkrahul/twitter-hate-speech). Due to this shortcoming, the project was never fully implemented into an actual moderator bot. 
