f = open ("train","r",encoding="utf8")
lines = f.readlines()

list_words= [] #a list of tweet_notation
for i in range(20000): #range(len(lines))
    if (lines[i]!=" "):      

        lines[i]= lines[i].replace("\n","")                       
        lines[i] =lines[i].split(" ") #word[i] = each object in the list, in a form of a list 'tweet','notation'

        if (lines[i]!=[""] and lines[i]!=[".",'O'] and lines[i]!=["","O"] and lines[i][1]!="") :
            list_words.append(lines[i]) #deletes lines with nothing


            
        
wordcount_dict={}
# print (list_words)

for j in range(len(list_words)): #len(list_words)
    word = list_words[j][0]
    sentiment= list_words[j][1]

    if word not in wordcount_dict:
        sentiment_count={sentiment:1}
        wordcount_dict[word]=sentiment_count
        
    else: #word is already a key in wordcount_dict
        if sentiment not in word:
            sentiment_count[sentiment]=1
            wordcount_dict[word]=sentiment_count
        else:
            sentiment_count = wordcount_dict[word]
            print (sentiment_count)
            sentiment_count[sentiment]+=1
            wordcount_dict[word]=sentiment_count
print(wordcount_dict)




