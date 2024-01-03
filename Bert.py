from transformers import pipeline

# text classification : 감정분석
classifier= pipeline('sentiment-analysis')
sentence = "아잉 너무좋아" #감성분석하고 싶은 문장 입력
result = classifier(sentence)[0]
print("입력문장: ",sentence)
print('감성분석결과: %s, 감성스코어: %0.2f'%(result['label'],result['score']))

# text generation : 문장 생성
text_generator = pipeline('text-generation')
sentence = "I was sitting on my bed."
result = text_generator(sentence)
print("생성된 문장: ",result[0]['generated_text'])

# Question answering : 기계독해
question_answering = pipeline("question-answering")
context = """
Shake Shack is an American fast casual restaurant chain based in New York City. It started out as a hot dog cart inside Madison Square Park in 2001, and its popularity steadily grew.
In 2004, it received a permit to open a permanent kiosk within the park, expanding its menu from New York–style hot dogs to one with hamburgers, hot dogs, fries and its namesake milkshakes.
Since its founding, it has been one of the fastest-growing food chains, eventually becoming a public company filing for an initial public offering of stock in late 2014. The offering priced on January 29, 2015; the initial price of its shares was at $21, immediately rising by 123% to $47 on their first day of trading.
Shake Shack Inc. owns and operates over 400 locations globally.
"""
question = "How many locations Shake Shack Inc. operates?"
result = question_answering(question=question,context=context)
print("지문: ", context)
print("문제: ",question)
print("답: ", result['answer'])

# Fill-mask : 빈칸 예측하기
from pprint import pprint
fill_mask = pipeline("fill-mask")
sentence =  f"AlphaGo is a computer {fill_mask.tokenizer.mask_token} that plays the board game Go."
result = fill_mask(sentence)
print("문장: ",sentence)
pprint(result)

# Summarization : 문서 요약하기
summerizer = pipeline('summarization')
article = """An apple is an edible fruit produced by an apple tree (Malus domestica). Apple trees are cultivated worldwide and are the most widely grown species in the genus Malus. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found today. Apples have been grown for thousands of years in Asia and Europe and were brought to North America by European colonists. Apples have religious and mythological significance in many cultures, including Norse, Greek, and European Christian tradition.
Apples grown from seed tend to be very different from those of their parents, and the resultant fruit frequently lacks desired characteristics. Generally, apple cultivars are propagated by clonal grafting onto rootstocks. Apple trees grown without rootstocks tend to be larger and much slower to fruit after planting. Rootstocks are used to control the speed of growth and the size of the resulting tree, allowing for easier harvesting.
There are more than 7,500 known cultivars of apples. Different cultivars are bred for various tastes and uses, including cooking, eating raw, and cider production. Trees and fruit are prone to a number of fungal, bacterial, and pest problems, which can be controlled by a number of organic and non-organic means. In 2010, the fruit's genome was sequenced as part of research on disease control and selective breeding in apple production.
Worldwide production of apples in 2018 was 86 million tonnes, with China accounting for nearly half of the total."""
result = summerizer(article,max_length=150,min_length=50)
print("문서: ",article)
print("요약: ",result[0]['summary_text'])