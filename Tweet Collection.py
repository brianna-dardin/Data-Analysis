# all the required imports for this script to work
import tweepy
import json
from pymongo import MongoClient
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore

# jump to if __name__ == "__main__" at the bottom before reading these 2 functions
# this is the main function that collects the tweets and inserts them into MongoDB
def collect_tweets():
    # make sure to download MongoDB and connect to localhost (default option)
    # then create a database called "tweet_db" and a collection/table called "tweets"
    # the database and collection names can be changed as long as you update them here
    # this code connects to the collection so we can insert the tweets
    client = MongoClient(host='localhost')
    db = client['tweet_db']
    collection = db['tweets']
    
    # this code assumes multiple search terms in a dictionary but will still work with 1 key
    for k in search_terms.keys():
        # call the search_tweets function (in a separate function only for readability)
        result = search_tweets(k,search_terms[k])
        # update the dictionary with the most recent ID searched
        search_terms[k] = result[0]
        # print how many tweets were found for this search term
        keyword = k.split()[0]
        print(keyword,len(result[1]))
        
        for t in result[1]:
            # load the JSON representation of the tweet data
            tweet = json.loads(json.dumps(t._json))
            # add the search keyword to the document for tracking
            tweet['keyword'] = keyword
            # insert the JSON tweet data into the MongoDB database
            collection.insert_one(tweet)
            
# adapted from https://stackoverflow.com/questions/22469713/managing-tweepy-api-search
# this function will collect tweets based on the given query
def search_tweets(query,since_id):
    # since_id is the ID of the most recent tweet that was previously processed
    # this code will search for new tweets posted since this one was posted
    # new_id is used to store the next value of since_id
    new_id = since_id
    # max_id is updated within the following loop to keep track of where we are
    # since Twitter returns the most recent tweets first 
    max_id = -1
    searched_tweets = []
    # in order to respect API limits we set a max number of tweets per query
    # this code repeats until we've hit this max number
    max_tweets = 1000
    while len(searched_tweets) < max_tweets:
        # Twitter API search is paginated so this keeps track of where we are in the results
        count = max_tweets - len(searched_tweets)
        try:
            # this is the Twitter API call using the query, making sure it's in English
            # count tells it which page to start from
            # max_id tells it which tweet to start from on that page
            # extended tweet mode ensures we receive the full text of each tweet
            new_tweets = api.search(q=query, lang='en', count=count, max_id=str(max_id - 1),\
                                    since_id=since_id, tweet_mode='extended')
            # stop execution if no tweets are found
            if not new_tweets:
                break
            # otherwise add these tweets to the main array
            searched_tweets.extend(new_tweets)
            # update new_id to be the most recent tweet ID
            if new_tweets[0].id > new_id:
                new_id = new_tweets[0].id
            # update max_id to the oldest tweet in this set for the next iteration
            max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # depending on TweepError.code, one may want to retry or wait
            # to keep things simple, we will give up on an error
            print(e)
            break
    return (new_id, searched_tweets)
      
if __name__ == "__main__":
    # define the terms of interest
    terms = ['#CaptainMarvel', '#WonderWoman']
    search_terms = {}
    for t in terms:
        # for each term, construct a query where they are mentioned by others
        # but exclude all retweets
        term = t + ' -filter:retweets'
        # add the query to the dictionary with a default ID to be used later
        search_terms[term] = -1
    
    # these keys and secrets are required by Twitter to use their API
    ckey = ''
    consumer_secret = ''
    access_token_key = ''
    access_token_secret = ''
    
    # this code sets up the connection to the Twitter API
    auth = tweepy.OAuthHandler(ckey, consumer_secret)
    auth.set_access_token(access_token_key, access_token_secret)
    api = tweepy.API(auth)
    
    # make sure to download MongoDB and connect to localhost (default option)
    # then create a database called "tweet_db" and a collection/table called "jobs"
    # the database and collection names can be changed as long as you update them here
    # this code sets up the scheduler that will run the collect_tweets function
    # here it will run every 2 hours but you can change this
    client = MongoClient(host='localhost')
    scheduler = BlockingScheduler()
    scheduler.add_jobstore(MongoDBJobStore(database='tweet_db',collection='jobs',client=client))
    scheduler.add_job(collect_tweets, 'interval', minutes=120)
    
    # this code will run in its own dedicated console until you manually stop execution
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass