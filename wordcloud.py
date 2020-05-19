from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def generate_wordcloud(filename):
    f = 0
    try:
        f = open(filename, 'r')
    except:
        print('file error')
        return
    text = f.read().replace('\n', ' ')

    stop_words = set(STOPWORDS)
    wc = WordCloud(background_color='White',
                    width = 800,
                    height = 800,
                    max_words = 200,
                    stopwords = stop_words)
    wc.generate(text)
    plt.imshow(wc)
    plt.axis('off')
    plt.savefig('out.png')
    plt.show()

generate_wordcloud('love_words.txt')