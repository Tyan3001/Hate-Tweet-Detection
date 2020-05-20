from Data_Handling import import_data


Data = import_data("Train.csv")
f1 = open('hate_words.txt', 'a')
f2 = open('love_words.txt', 'a')

for index, row in Data.iterrows():
    if str(row['label']) == '1':
        for i in row['Tweet_words']:
            try:
                f1.write(i + '\n')
            except:
                pass
    else:
        for i in row['Tweet_words']:
            try:
                f2.write(i + '\n')
            except:
                pass
f1.close()
f2.close()