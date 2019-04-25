#! usr/bin/env python3
import praw
import pandas as pd
import datetime as dt
import time




reddit = praw.Reddit(client_id='#', \
					 client_secret='#-#', \
					 user_agent='#', \
					 username='#', \
					 password='#')

def srapeSubreddit(subName):
	f = open("../data/"+subName+".txt","w")
	f.write("test")
	subreddit = reddit.subreddit(subName)
	#top_subreddit = subreddit.top()
	top_subreddit = subreddit.top(limit=500)

	topics_dict = { "title":[], 
				"score":[], \
				"id":[], "url":[], 
				"comms_num": [], 
				"created": [], 
				"body":[]}
	for submission in top_subreddit:
		docString = ""
		topics_dict["title"].append(submission.title)
		docString+= "<'|'>"+str(str(submission.title).translate(str.maketrans(' ',' ', "(,),\',\",!,@,#,$,%,^,&,*,{,},{,},-,_,=,+,;,:,,.,<,>,/,?,\\,|,`,~,\n")))
		topics_dict["score"].append(submission.score)
		docString+= "<'|'>"+str(str(submission.score).translate(str.maketrans(' ',' ', "(,),\',\",!,@,#,$,%,^,&,*,{,},{,},-,_,=,+,;,:,,.,<,>,/,?,\\,|,`,~,\n")))
		topics_dict["id"].append(submission.id)
		docString+= "<'|'>"+str(str(submission.id).translate(str.maketrans(' ',' ', "(,),\',\",!,@,#,$,%,^,&,*,{,},{,},-,_,=,+,;,:,,.,<,>,/,?,\\,|,`,~,\n")))
		topics_dict["url"].append(submission.url)
		docString+= "<'|'>"+str(str(submission.url))
		topics_dict["comms_num"].append(submission.num_comments)
		docString+= "<'|'>"+str(str(submission.num_comments).translate(str.maketrans(' ',' ', "(,),\',\",!,@,#,$,%,^,&,*,{,},{,},-,_,=,+,;,:,,.,<,>,/,?,\\,|,`,~,\n")))
		topics_dict["created"].append(submission.created)
		docString+= "<'|'>"+str(str(submission.created).translate(str.maketrans(' ',' ', "(,),\',\",!,@,#,$,%,^,&,*,{,},{,},-,_,=,+,;,:,,.,<,>,/,?,\\,|,`,~,\n")))
		topics_dict["body"].append(submission.selftext)
		docString+= "<'|'>"+str(str(submission.selftext).replace("\n"," "))+"<'|'>\n"
		f.write(docString)
	f.close()

	#topics_data = pd.DataFrame(topics_dict)
	'''titleList = topics_dict["title"]
	bodyList = topics_dict["body"]
	for title in titleList:
		print(title)
	print("\n\n------------------------------------------------------------------------------------------\n\n")
	for body in bodyList:
		print(body)
	for i in range(len(topics_dict["id"])):

		for key in topics_dict:
			print(topics_dict[key][i])'''

	'''
for item in topics_dict:
	print(topics_dict[item])'''


def main():
	start = time.time()
	dotWord = "...................................................."
#----------------------------------------------------------------------------------------------------------
	statement = "libertarian"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)

	srapeSubreddit(statement)

	end1 = time.time()
	print("DONE \t",round(end1 - start,15))
#-----------------------------------------------------------------------------------------------------------
	statement = "the_donald"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	
	srapeSubreddit(statement)

	end1 = time.time()
	print("DONE \t",round(end1 - start,15))

#-----------------------------------------------------------------------------------------------------------
	statement = "liberal"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	
	srapeSubreddit(statement)

	end1 = time.time()
	print("DONE \t",round(end1 - start,15))

#-----------------------------------------------------------------------------------------------------------
	statement = "greenparty"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	
	srapeSubreddit(statement)

	end1 = time.time()
	print("DONE \t",round(end1 - start,15))
#-----------------------------------------------------------------------------------------------------------
	statement = "socialism"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	
	srapeSubreddit(statement)

	end1 = time.time()
	print("DONE \t",round(end1 - start,15))
#-----------------------------------------------------------------------------------------------------------
	statement = "communism"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)

	srapeSubreddit(statement)

	end1 = time.time()
	print("DONE \t",round(end1 - start,15))
#-----------------------------------------------------------------------------------------------------------
	statement = "democrats"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	
	srapeSubreddit(statement)

	end1 = time.time()
	print("DONE \t",round(end1 - start,15))
#-----------------------------------------------------------------------------------------------------------	
	statement = "republican"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	
	srapeSubreddit(statement)

	end1 = time.time()
	print("DONE \t",round(end1 - start,15))
#-----------------------------------------------------------------------------------------------------------	
	statement = "progun"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	
	srapeSubreddit(statement)

	end1 = time.time()
	print("DONE \t",round(end1 - start,15))



main()
