# all the required imports for this script to work
import pandas as pd
from pymongo import MongoClient
import re
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob

def main():
    # connect to MongoDB database
    client = MongoClient(host='localhost')
    db = client['tweet_db']
    collection = db['tweets']
    
    # Expand the cursor and construct the DataFrame
    cursor = collection.find()
    data = list(cursor)
    df = pd.DataFrame(data)
    
    new_df = pd.DataFrame()
    
    # these are the only fields we are interested in from the top level object
    base_list = ['id','keyword','full_text','retweet_count','favorite_count']
    for b in base_list:
        new_df[b] = df[b]
    
    # these are the only fields we are interested in from the user object
    user_list = ['location','verified','followers_count']
    for u in user_list:
        new_df[u] = df.loc[:, 'user'].apply(lambda x: x[u])
        
    # I forgot to add the search hashtag as a field on the older tweets so this assigns
    # tweets without it the appropriate keyword    
    new_df['keyword'] = new_df.apply(keyword,axis=1)
    
    # some duplicates were picked up during the collection process
    new_df.drop_duplicates(inplace=True)
    
    # this creates a new field with clean versions of the tweets for analysis
    new_df['clean_text'] = new_df.apply(lambda x: process_text(x['full_text']),axis=1)

    # using the clean tweets, create a corpus for both hashtags
    new_dict = dict.fromkeys(new_df['keyword'].unique(),'')
    for i, row in new_df.iterrows():
        new_dict[row['keyword']] += row['clean_text'] + " "
    
    for k in new_dict.keys():
        # print a list of the most common words for each hastag
        print(k)
        text = new_dict[k].strip()
        words = Counter(text.split(" "))
        most_common = words.most_common(52)
        print(most_common)
        
        # generate the word cloud
        wordcloud = WordCloud(background_color="white").generate(text)
        wordcloud.to_file(k[1:]+".png")
    
    # this assigns each tweet a sentiment score between -1 and 1    
    new_df['sentiment'] = new_df.apply(lambda x: sentiment(x['clean_text']),axis=1)

    # finally create an excel file for manual analysis
    new_df.to_excel('full_dataset.xlsx', index=False, encoding='utf-8')

# this function priortizes assigning tweets to #WonderWoman since there aren't as many
def keyword(x):
    if pd.isna(x['keyword']):
        lower = x['full_text'].lower()
        for w in dc_words:
            if w in lower:
                return '#WonderWoman'
        return '#CaptainMarvel'
    else:
        return x['keyword']
        
def process_text(proc):
    #in case the text has this character, make sure it's replaced with a space 
    #so word boundaries are respected
    proc = proc.replace("\xa0"," ")
    
    #remove URLs
    proc = re.sub(r'http\S+', '',proc)
    
    #remove all @ mentions for anonymity
    proc = re.sub(r'@[\w]*', '', proc)
    
    #remove emojis
    proc = proc.encode('ascii', 'ignore').decode('ascii')
    
    #remove stopwords
    proc = [i.lower() for i in wordpunct_tokenize(proc) if i.lower() not in stop_words]
    proc = " ".join(proc)
    
    #some articles have 2 letter abbreviations separated with periods (a.m., U.S.)
    #this finds them and removes the periods so they can be considered as n-grams
    #there is likely a better way of doing this but I tried
    match = re.findall(r'[A-Za-z]\.[A-Za-z]\.',proc)
    if(len(match)>0):
        for m in match:
            proc = proc.replace(m, m.replace('.',''))
            
    #This removes periods from acronyms longer than 2 characters
    proc = re.sub(r'(?<!\w)([A-Z])\.', '',proc)
            
    #This removes numbers and some special characters
    proc = re.sub('[^A-Za-z.?: ]+', '', proc)
    
    #There is probably a more elegant way of doing this, but this ensures that sentence
    #boundaries are respected by putting in a space instead (otherwise the words combine)
    #However this operation can introduce extra whitespace which needs to be removed as well
    proc = re.sub(r'[.?:]+', " ",proc)
    proc = re.sub("\s\s+" , " ", proc)
    proc = proc.strip()
    
    return proc

# TextBlob makes it very easy to get a sentiment score
def sentiment(x):
    blob = TextBlob(x)
    return blob.sentiment.polarity

if __name__ == "__main__":
    # lists for both DC and Marvel needed to be created in order to filter these words out
    # otherwise the word clouds would be dominated by these terms
    # the words also don't contribute anything meaningful to the sentiment analyzer
    dc_words = ['wonderwoman', 'wonder woman', 'batman', 'shazam', 'dccomics', 'superman',
                'justice league', 'justiceleague', 'dcmovies', 'theflash', 'gal gadot','gal',
                'galgadot', 'aquaman', 'the flash', 'dceu', 'dc', 'gotham','wonder','ww',
                'diana','dcuniverse','gadot', '#wonderwoman','#WonderWoman']
    
    mcu_words = ['captainmarvel','captain marvel','avengers','endgame','avengersendgame',
                 'thor','ironman','iron man','captainamerica', 'captain america','marvel',
                 'blackwidow','black widow','blackpanther','black panther','hulk','antman',
                 'ant man','loki','valkyrie','nick fury','thanos','gamora','gotg','starlord',
                 'rocket raccoon','groot','drax','spiderman','tony stark','steve rogers',
                 'tonystark','steverogers','hawkeye','nebula','rocketraccoon','warmachine',
                 'captain','okoye','thevalkyrie','mcu','wong','pepperpotts','korg','miek',
                 'marvelstudios','doctorstrange','brielarson','brie larson','happyhogan',
                 'mbaku','scarletwitch','brie','larson','nick fury','nickfury','mantis',
                 'thewasp','shuri','caroldanvers','wintersoldier','dontspoiltheendgame',
                 'carol','drstrange','buckybarnes']
    
    # create the stopword list and update it to remove common words that would obscure
    # the presence of sentiment words we want to analyze
    stop_words = stopwords.words('english')
    stop_words.extend(['woman','movie','movies','superhero','superheroes','comics','comic',
                       'comicbooks','women','hero','film','amp'])
    stop_words.extend(dc_words)
    stop_words.extend(mcu_words)
    
    main()