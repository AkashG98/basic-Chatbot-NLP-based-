import nltk
import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading popular packages
nltk.download('punkt') 
nltk.download('wordnet') 



import numpy as np
import random
import string 


f=open('chatbot.txt','r',errors='ignore')
raw=f.read()
raw=raw.lower()

sent_tokens=nltk.sent_tokenize(raw)#stores all tokenized sentences
word_tokens=nltk.word_tokenize(raw)#stores all tokenized words


lemmer=WordNetLemmatizer()

def LemTokens(tokens):
    return[lemmer.lemmatize(token) for token in tokens]


remove_punc_dict=dict((ord(punct),None)for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

Greetings_in=["hello","hi","greetings","sup","what's up","hey",]
Greetings_out=["hi","hey","nods","hi! there","hello","I am glad ! you are talking to me"] 

def greetings(sentence):
    for word in sentence.split():
        if word.lower() in Greetings_in:
            return random.choice(Greetings_out)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def response(user_resp):
    robo_resp=''
    sent_tokens.append(user_resp)
    TfidfVec=TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')#gives similarity between chatbot and user response
    tfidf=TfidfVec.fit_transform(sent_tokens)
    vals=cosine_similarity(tfidf[-1],tfidf)#sparse matrix that computes similarity between input and output
    #the basic purpose for the above function is to find a similar word to our input from bag of words
    
    idx=vals.argsort()[0][-2] #second index taken so that user input does not come into consideration
    flat=vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]
    if(req_tfidf==0):#in case no word similar to our input word is found in the bag of words
        robo_resp=robo_resp+"Sorry I didn't get you"
        return robo_resp
    else:
        robo_resp=robo_resp+sent_tokens[idx] #in case word matches add to response system and generate output
        return robo_resp
    


flag=True
print("Hello User. I can tell anything regarding chatbots. If you want to exit,type Bye!")


while(flag==True):
    user_response=input()
    user_response=user_response.lower()
    if(user_response!="bye!"):
        if(user_response=='thanks' or user_response=='thank you'):
            flag=False
            print("TATA.")
        else:
            if(greetings(user_response)!=None):
                print("\n"+greetings(user_response))
            else:
                print(end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("Bye! take care..")
        
            
            
    
    
    
    




