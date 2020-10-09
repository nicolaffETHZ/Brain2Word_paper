import numpy as np
import os
import pandas as pd
import scipy.io as sio
import encoder

def load_words():
    path = str(os.path.dirname(os.path.abspath(__file__))) + '/data/subjects/P01/examples_180concepts_sentences.mat'
    mat_file = sio.loadmat(path)
    words = mat_file['keyConcept']  
    return words

def checker(string):
    string = string.replace("'ve",'')
    string = string.replace("@",'')
    string = string.replace("'re",'')
    string = string.replace("malfoy'll",'malfoy')
    string = string.replace("'d",'')
    string = string.replace("?",'')
    string = string.replace("'s",'')
    string = string.replace(":",'')
    string = string.replace("!",'')
    string = string.replace('"','')
    string = string.replace(".",'')
    string = string.replace("--",'')
    string = string.replace("'",'')
    string = string.replace(",",'')
    string = string.replace(';','')
    string = string.replace('‘','')
    string = string.replace('(','')
    string = string.replace(')','')
    string = string.replace('\'','')
    string = string.replace(' ','')
    return(string)

def files_exist():
    booler = os.path.exists(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/converter_table.npy')
    booler = booler and os.path.isfile(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/top_ten_embeddings')
    booler = booler and os.path.isfile(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/word_sets_num.npy')
    booler = booler and os.path.isfile(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/word_sets.txt')
    return booler

def generate_look_ups():
    if  not os.path.exists(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/converter_table.npy'):
        converter_table()
    if not os.path.isfile(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/top_ten_embeddings'):
        top_ten_embeddings()
    if not os.path.isfile(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/word_sets.txt'):
        print('First create a text file with top 5 words you want the generation to anchored with.')
    else:
        if not os.path.isfile(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/word_sets_num.npy'):
            word_sets()


def word_sets():

    lister = []
    words = load_words()
    for i in range(180):
        lister.append(words[i][0][0])

    
    file1 = open(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/word_sets.txt',"r+")  
    length = len(file1.readlines())
    file1 = open(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/word_sets.txt',"r+")  
    holder = np.zeros((length,5),dtype=int)
    for i in range(length):
        line = file1.readline()
        lines = line.split(',')
        for j in range(5):
            if j == 0:
                numb = lister.index(lines[j].strip())
            else:
                numb = lister.index(lines[j].strip())
            holder[i,j] = numb
    np.save(file=str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/word_sets_num.npy', arr=holder) 

def converter_table():
    path = str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/converter_table'

    embeddings_dict = {}
    with open(str(os.path.dirname(os.path.abspath(__file__))) + "/glove.42B.300d.txt", 'r') as f:    #Need glove embeddings dataset!
        for line in f:
            try:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
            except ValueError:
                print(line.split()[0])

    model_name='124M'
    models_dir= str(os.path.dirname(os.path.abspath(__file__))) + '/models_gpt'
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)

    holder = np.zeros((50257,300))

    for i in range(50257):
        
        try:
            word = enc.decode([i])
            word = checker(word.strip().lower())
            glove = embeddings_dict[word]
            holder[i,:] = glove
        except:
            word = enc.decode([i])
            holder[i,:] = np.zeros((300)) + 500


    np.save(file=path, arr=holder) 
    print('Converter table was generated')

def find_closest_embeddings_cosine_prox(embedding,embeddings_dict):
    return sorted(embeddings_dict.keys(), key=lambda word: cosine(embeddings_dict[word], embedding))

def top_ten_embeddings():

    path = str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/top_ten_embeddings_two'

    word = load_words()
    words = load_words_and_glove()

    embeddings_dict = {}
    with open(str(os.path.dirname(os.path.abspath(__file__))) + "/glove.42B.300d.txt", 'r') as f:    #Need glove embeddings dataset!
        for line in f:
            try:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
            except ValueError:
                print(line.split()[0])
    embbeddings = []

    for i in range(words.shape[0]):
        top_ten = find_closest_embeddings_cosine_prox(words[i,:],embeddings_dict)[:11]
        embbeddings.append(top_ten)
    df = pd.DataFrame(embbeddings)
    df.to_csv(path, index= False)


def isSubset(subarraies, array):
    counter = 0
    for subarray in subarraies:
        for i in range(len(array)):
            for j in range(len(subarray)):
                if i+j<len(array):
                    if array[i+j] == subarray[j]:
                        if j == len(subarray)-1:
                            counter+=1
                    else:
                        break
    return counter

def tokens_from_words(Numbers):
    model_name='124M'
    models_dir= str(os.path.dirname(os.path.abspath(__file__))) + '/models_gpt'
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)
    word = load_words()
    container = []
    for Nr in Numbers:
        container.append(enc.encode(' ' + word[Nr][0][0]))
    return container

def load_words_and_glove():
    path = str(os.path.dirname(os.path.abspath(__file__))) + '/data/glove_data/180_concepts.mat'
    mat_file = sio.loadmat(path)
    glove =mat_file['data']
    return  glove

def related_words():
    top_ten = pd.read_csv(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/top_ten_embeddings')
    model_name='124M'
    models_dir=str(os.path.dirname(os.path.abspath(__file__))) + '/models_gpt'
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))

    enc = encoder.get_encoder(model_name, models_dir)
    container = []
    for i in range(180):
        intermediate = []
        for j in range(11):
            shape = enc.encode(' ' + top_ten.iloc[i,j])
            intermediate.append(shape)
        container.append(intermediate)
    return container

def Harry_sentences_no_capital(counter, Sent_num):

    harry_dir = str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/words_fmri.npy'
    namelist = ['Harry', 'Ron', 'Malfoy', 'Neville', 'Dumbledore', 'Hermione', 'Potter', 'Weasley.', ' Potter',  'Potter,', ' Potter,']
    harry = np.load(harry_dir)
    total = ''
    q = 0
    while q < Sent_num:
        booler = False
        first = True
        stringe = ''
        for i in range(harry.size):
            j = counter + i
            if not first:
                booler = any(letter.isupper() for letter in harry[j]) or booler   
            else:
                booler = harry[j] in namelist or booler
            stringe = stringe + ' ' + harry[j]
            first = False
            if harry[j].__contains__('.'):
                counter = j+1
                break
        if not booler:
            q+=1
            total = total + ' ' + stringe
    return(total, counter)