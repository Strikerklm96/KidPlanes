from nltk import WordNetLemmatizer
from nltk import SnowballStemmer
wnl = WordNetLemmatizer()
stem = SnowballStemmer("english")

def shorten(nonsense):
    #return wnl.lemmatize(nonsense)
    return stem.stem(nonsense)


print(shorten("people"))
print(shorten("peoples"))
print(shorten("person"))
print(shorten("car"))
print(shorten("cars"))
print(shorten("generalized"))
print(shorten("generalization"))
print(shorten("general"))

phrase = "Hello, this is a string. Workers work on cars?"
originalPhrase = phrase

splits = ",;" + ".?!"  # divide words better
for c in splits:
    phrase = phrase.replace(c, " " + c)

#  TODO need to remove garbage symbols like < and ()

phrase = phrase.split()  # divide into words

for index, word in enumerate(phrase):  # simplify words
    phrase[index] = shorten(word)

print("Original:")
print(originalPhrase)
print("New:")
print(phrase)





