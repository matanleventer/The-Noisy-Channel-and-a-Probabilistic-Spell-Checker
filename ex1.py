import math
from collections import Counter
import random
import re

class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self,  lm=None,error_tables=None):
        """Initializing a spell checker object with a language model as an
        instance  variable.

        Args:
            lm: a language model object. Defaults to None.
        """
        self.lm = lm
        self.error_tables = error_tables


    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)

            Args:
                lm: a Spell_Checker.Language_Model object
        """
        self.lm = lm



    def add_error_tables(self, error_tables):
        """ Adds the specified dictionary of error tables as an instance variable.
            (Replaces an older value dictionary if set)

            Args:
            error_tables (dict): a dictionary of error tables in the format
            of the provided confusion matrices:
            https://www.dropbox.com/s/ic40soda29emt4a/spelling_confusion_matrices.py?dl=0
        """
        self.error_tables = error_tables


    def evaluate_text(self, text):
        """Returns the log-likelihood of the specified text given the language
            model in use. Smoothing should be applied on texts containing OOV words
    
           Args:
               text (str): Text to evaluate.
    
           Returns:
               Float. The float should reflect the (log) probability.
        """
        return self.lm.evaluate_text(text)

    def get_candidate_2(self,word):
        dict_candidate_all = {}
        dict_candidate = self.get_candidate(word,1)
        for c in dict_candidate.items():
            dict_candidate_2 = self.get_candidate(c[0][0],2)
            dict_candidate_all.update(dict_candidate_2)
        dict_candidate_all.update(dict_candidate)
        return dict_candidate_all

    def get_candidate(self,word,n):
        list_chars_engish =['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        error_types = ['insertion', 'deletion', 'substitution', 'transposition']
        len_word = len(word)
        dict_cand = {}
        if n==1:
            bool=True
        else:
            bool=False
        for e in error_types:
            if e == 'insertion':
                for c_e in list_chars_engish:
                    for l in range(len_word+1):
                        res = word[:l] + c_e + word[l:]
                        if res in self.lm.model_dict_1.keys() or bool:
                            if l == 0 :
                                dict_cand[((res,e,word,n))] = '#'+c_e
                            else:
                                dict_cand[((res, e,word,n))] = res[l-1] + c_e

            elif e == 'deletion':
                for l in range(len_word):
                    res = word[:l] + word[l+1:]
                    if res in self.lm.model_dict_1 or bool:
                        if l == 0:
                            dict_cand[((res, e,word,n))] = '#' + word[l]
                        else:
                            dict_cand[((res, e,word,n))] = word[l-1] + word[l]

            elif e == 'substitution':
                for c_e in list_chars_engish:
                    for l in range(len_word):
                        res = word[:l]+ c_e + word[l+1:]
                        if (res in self.lm.model_dict_1 or bool)and res !=word :
                            dict_cand[((res, e,word,n))] = word[l] + c_e

            else:
                for l in range(len_word-1):
                    res = word[:l] + word[l + 1] + word[l] + word[l + 2:]
                    if (res in self.lm.model_dict_1 or bool)and res !=word :
                        dict_cand[((res, e,word,n))] = word[l + 1] + word[l]

        return dict_cand


    def check_all_words(self,sentence):
        for loc, w in enumerate(sentence):
            if w in self.lm.model_dict_1.keys():
                continue
            else:
                return w
        return True

    def get_candidate_chars(self,word):
        list_chars_engish = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                             's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        error_types = ['substitution', 'transposition']
        len_word = len(word)
        dict_cand = {}
        for e in error_types:
            if e == 'substitution':
                for c_e in list_chars_engish:
                    for l in range(len_word):
                        res = word[:l] + c_e + word[l + 1:]
                        if (res in self.lm.model_dict_1 or bool) and res != word:
                            r=''
                            if word[l] == ' ':
                                r += '#'
                            else:
                                r  += word[l]
                            if c_e == ' ':
                                r += '#'
                            else:
                                r  += c_e

                            dict_cand[((res, e))] = r
            else:
                for l in range(len_word - 1):
                    res = word[:l] + word[l + 1] + word[l] + word[l + 2:]
                    if (res in self.lm.model_dict_1 or bool) and res != word:
                        r = ''
                        if word[l] == ' ':
                            r += '#'
                        else:
                            r += word[l]
                        if c_e == ' ':
                            r += '#'
                        else:
                            r += c_e
                        dict_cand[((res, e))] = r
        return dict_cand


    def spell_check_chars(self, text, alpha=0.95):
        text = normalize_text(text)
        fool = False
        if text[-1]=='.':
            fool =True
            text=text[:-1]
        text = text.replace(". ", ".")
        list_sentence = text.split('.')
        gen =''
        for sen in list_sentence:
            dict_cand = {}
            words = [sen[i:i+self.lm.n:] for i in range(len(sen)-self.lm.n+1)]
            cool= True
            for loc,j in enumerate(words):
                r_n = ''
                r_n_1 = ''
                for i in (range(self.lm.n)):
                    r_n += j[i] + '_'
                    if i != self.lm.n - 1:
                        r_n_1 += j[i] + '_'
                r_n = r_n[:-1]
                r_n_1 = r_n_1[:-1]
                if r_n in self.lm.model_dict.keys():
                    continue
                else:
                    cool = False
                    break
            if not cool:
                dict_candidate = self.get_candidate_chars(words[loc])
                for cand,action in dict_candidate:
                    new_sen = sen[:loc] + cand[0] + sen[loc+1:]
                    try:
                        dict_cand[new_sen] = (self.error_tables[action][dict_candidate[((cand,action))]] / self.lm.model_dict_n_1[r_n_1]) * self.lm.evaluate_text_chars(r_n,True) * (1-alpha)
                    except:
                        continue
            non_zero_dict = {k: v for k, v in dict_cand.items() if v != 0}
            if len(non_zero_dict) == 0:
                gen += sen
            else:
                max_word = max(non_zero_dict, key=non_zero_dict.get)
                max_value = non_zero_dict[max_word]
                if self.lm.evaluate_text_chars(words[loc],True) * alpha < max_value:
                    gen += max_word
                else:
                    gen += sen
            gen += '.'
        if fool:
            gen += '.'
        return gen[:-1]



    def spell_check(self, text, alpha):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """
        if self.lm.chars:
            return self.spell_check_chars(text, alpha)
        text = normalize_text(text)
        text = text.replace(". ", ".")
        cool = False
        w_bad=''
        if text[-1]=='.':
            cool =True
            text=text[:-1]
        elif text[-1]==' ':
            text=text[:-1]
        list_sentence = text.split('.')
        new_text = ''
        for sen in list_sentence:
            dict_candidate_res_0 = {}
            dict_candidate_res_1 = {}
            dict_candidate_res_2 = {}
            se = sen.split(' ')
            if se[0] == '':
                se=se[1:]
            result  = self.check_all_words(se)
            if result != True:
                w_bad = result
                result=False
            for loc,w in enumerate(se):
                if w == w_bad or result:
                    dict_candidate_all = self.get_candidate_2(w)
                    for key, val in dict_candidate_all.items():
                        cand = key[0]
                        method = key[1]
                        old_word = key[2]
                        number_time = key[3]
                        if val not in self.error_tables[method]:
                            continue
                        nom_1 = self.error_tables[method][val]
                        if method == 'substitution':
                            denom_1 = self.lm.model_char[val[1]] + 1
                        else:
                            denom_1 = self.lm.model_chars[val] + 1
                        p_x_w = nom_1 / denom_1
                        correct_se = se[:loc] + [cand] + se[loc+1:]
                        correct_se = ' '.join(correct_se)
                        correct_old = se[:loc] + [old_word] + se[loc + 1:]
                        correct_old = ' '.join(correct_old)
                        if len(se) < self.lm.n:
                            if number_time == 1:
                                dict_candidate_res_1[((correct_se, cand))] = ((p_x_w, self.lm.evaluate_text(cand,True)))
                            else:
                                dict_candidate_res_2[((correct_se, correct_old, cand, old_word))] = ((p_x_w, self.lm.evaluate_text(cand,True)))
                        else:
                            if number_time == 1:
                                dict_candidate_res_1[((correct_se,cand))] = ((p_x_w  , self.lm.evaluate_text(correct_se,True)))
                            else:
                                dict_candidate_res_2[((correct_se,correct_old,cand,old_word))] = ((p_x_w  , self.lm.evaluate_text(correct_se,True)))

            for key in dict_candidate_res_2.keys():
                if key[0] not in dict_candidate_res_1.keys() and key[0]!=sen and key[2] in self.lm.model_dict_1:
                    value = dict_candidate_res_2[key][0] * dict_candidate_res_2[key][1] * dict_candidate_res_1[((key[1],key[3]))][0] * (1-alpha)
                    if key[0] in dict_candidate_res_0:
                        if value > dict_candidate_res_0[key[0]]:
                            dict_candidate_res_0[key[0]] =value
                    else:
                        dict_candidate_res_0[key[0]] = value

            for key in dict_candidate_res_1.keys():
                if key[0] !=sen and key[1] in self.lm.model_dict_1:
                    value =  dict_candidate_res_1[key][0] * dict_candidate_res_1[key][1] * (1-alpha)
                    if key[0] in dict_candidate_res_0:
                        if value > dict_candidate_res_0[key[0]]:
                            dict_candidate_res_0[key[0]] =value
                    else:
                        dict_candidate_res_0[key[0]] = value

            non_zero_dict = {k: v for k, v in dict_candidate_res_0.items() if v != 0}
            if len(non_zero_dict) == 0:
                new_text += sen
            else:
                max_word = max(non_zero_dict, key=non_zero_dict.get)
                max_value = non_zero_dict[max_word]
                if len(se) < self.lm.n:
                    if w_bad =='':
                        new_text += sen
                    elif self.lm.evaluate_text(w_bad,True) * alpha < max_value:
                        new_text += max_word
                    else:
                        new_text += sen
                else:
                    if self.lm.evaluate_text(sen,True) * alpha  < max_value:
                        new_text += max_word
                    else:
                        new_text += sen
            new_text += '. '
        if cool:
            new_text= new_text[:-1]
        else:
            new_text = new_text[:-2]
        return new_text


    #####################################################################
    #                   Inner class                                     #
    #####################################################################

    class Language_Model:
        """The class implements a Markov Language Model that learns a model from a given text.
            It supports language generation and the evaluation of a given string.
            The class can be applied on both word level and character level.
        """

        def __init__(self, n=3, chars=False):
            """Initializing a language model object.
            Args:
                n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                chars (bool): True iff the model consists of ngrams of characters rather than word tokens.
                              Defaults to False
            """
            self.n = n
            self.chars = chars
            self.model_dict = Counter()
            self.model_dict_n_1 = Counter()
            self.model_dict_1 = Counter()

            self.model_chars = Counter()
            self.model_char = Counter()#a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
            #NOTE: This dictionary format is inefficient and insufficient (why?), therefore  you can (even encouraged to)
            # use a better data structure.
            # However, you are requested to support this format for two reasons:
            # (1) It is very straight forward and force you to understand the logic behind LM, and
            # (2) It serves as the normal form for the LM so we can call get_model_dictionary() and peek into you model.

        def build_model(self, text):  # should be called build_model
            """populates the instance variable model_dict.

                Args:
                    text (str): the text to construct the model from.
            """
            text = normalize_text(text)

            self.model_dict = self.build_model_1(text,self.n)
            self.model_dict_n_1 = self.build_model_1(text,self.n-1)
            self.model_dict_1 = self.build_model_1(text,1)
            self.model_chars = self.build_model_2(text,2)
            self.model_char = self.build_model_2(text,1)
            if self.chars:
                self.model_dict_1 = self.model_char

        def build_model_1(self, text, n):  # should be called build_model
            """populates the instance variable model_dict.

                Args:
                    text (str): the text to construct the model from.
            """
            if not self.chars:
                if n==1:
                    sentences = text.replace('.','').split(' ')
                else:
                    sentences = text.split('.')
                # Add padding of 2 words to each sentence in the list
                padded_sentences = []
                for sentence in sentences:
                    if sentence == '':
                        continue
                    words = sentence.strip().split()  # strip removes leading/trailing whitespace
                    padded_words = ['<PAD>'] * (n-1) + words + ['<PAD>'] * (n-1)
                    padded_sentence = ' '.join(padded_words)
                    padded_sentences.append(padded_sentence)

                # Join padded sentences back into a full sentence
                padded_string = ' '.join(padded_sentences)
                padded_string = padded_string.split(' ')
                com_words = []
                for i in range(len(padded_string) - n + 1):
                    r = ''
                    for num in range(n):
                        if n == num + 1:
                            r += padded_string[num + i]
                        else:
                            r += padded_string[num + i] + '_'
                    com_words.append(r)
            else:
                text_ = ''.join(re.findall(r'[a-z .]', text))
                com_word=[]
                com_words=[]
                for i in range(len(text_) - self.n + 1):
                    com_word.append(text_[i:i + n])
                for i in com_word:
                    r=''
                    count = 0
                    for j in i:
                        if j=='.':
                            while count != n:
                                r+='<PAD>_'
                                count += 1
                            break
                        else:
                            count+=1
                            r+=j+'_'
                    r=r[:-1]
                    com_words.append(r)
            return Counter(com_words)


        def build_model_2(self, text,num):  # should be called build_model
            """populates the instance variable model_dict.

                Args:
                    text (str): the text to construct the model from.
            """
            text_ = text.replace(' ', '#')
            text_ = text_.replace('.', '#')
            text_ = text_.replace('##', '#')
            text_ = ''.join(re.findall(r'[a-z#]', text_))
            text_ = '#' + text_
            counts = Counter(text_[i:i + num] for i in range(len(text_) - self.n + 1))
            return counts

        def candidate(self,gen,old=''):
            dict_cand = {}
            number = min(self.n,3)
            for word in self.model_dict.keys():
                gen_new = gen
                while number>0:
                    try:
                        if gen_new == word.split('_')[:(len(gen_new))]:
                            if (word.split('_')[len(gen_new):])[0] not in dict_cand:
                                dict_cand[(word.split('_')[len(gen_new):])[0]] = self.model_dict[word]* 15 ** len(gen_new)
                            else:
                                dict_cand[(word.split('_')[len(gen_new):])[0]] += self.model_dict[word]* 15 ** len(gen_new)
                                break
                    except:
                        break
                    gen_new = gen_new[1:]
                    number=-1
            if old in dict_cand:
                del dict_cand[old]
            if (len(dict_cand) == 0):
                words = [key for key in self.model_dict.keys() if key.startswith(('<PAD>_'*(self.n-1))[:-1])]
                if (('<PAD>_'*(self.n))[:-1]) in words:
                    words.remove(('<PAD>_'*(self.n))[:-1])
                weights = [self.model_dict[key] for key in words]
                return random.choices(words, weights=weights)[0].split('_')[-1]
            words = list(dict_cand.keys())
            weights = list(dict_cand.values())
            random_word = random.choices(words, weights=weights)[0]
            return random_word

        def get_model_dictionary(self):
            """Returns the dictionary class object
            """
            return self.model_dict

        def get_model_window_size(self):
            """Returning the size of the context window (the n in "n-gram")
            """
            return self.n


        def generate_chars(self, context=None, n=20):
            context = normalize_text(context)
            context_split = [context[i:i + self.n] for i in range(len(context) - self.n + 1)]
            len_context_split = len(context_split)
            gen = context
            if len_context_split >= n:
                return ''.join(context[:20])
            number_of_words = n - len_context_split
            count = 0
            while count < number_of_words:
                words = list(self.model_dict.keys())
                weights = list(self.model_dict.values())
                random_word = random.choices(words, weights=weights)[0]
                split_random = random_word.split('_')
                for j in split_random:
                    if j=='<PAD>':
                        break
                r = ''.join(split_random)
                r = r.replace('_','')
                gen += r
                count += 1
            return gen


        def generate(self, context=None, n=20):
            """Returns a string of the specified length, generated by applying the language model
            to the specified seed context. If no context is specified the context should be sampled
            from the models' contexts distribution. Generation should stop before the n'th word if the
            contexts are exhausted. If the length of the specified context exceeds (or equal to)
            the specified n, the method should return a prefix of length n of the specified context.

                Args:
                    context (str): a seed context to start the generated string from. Defaults to None
                    n (int): the length of the string to be generated.

                Return:
                    String. The generated text.

            """
            if self.chars:
                return self.generate_chars(context,n)
            context = normalize_text(context)
            context = context.replace(". ", ".")
            context_split = context.split(' ')
            if context_split[0] == '':
                if len(context_split) == 1:
                    len_context_split = 0
                    gen = ''
                else:
                    context_split = context_split[1:]
                    gen = context + ' '
                    len_context_split = len(context_split)
            else:
                len_context_split = len(context_split)
                gen = context + ' '
            if len_context_split >= n:
                return ' '.join(context_split[:20])
            if len_context_split - self.n + 1 >= 0:
                context_split_ = context_split[len_context_split- self.n + 1:]
            else:
                context_split_ = context_split
            number_of_words = n - len_context_split
            count=0
            if len_context_split>0:
                old = context_split[-1]
            else:
                old=''
            while count <  number_of_words :
                gen_word = self.candidate(context_split_,old)

                if gen_word =='':
                    for i in range(self.n):
                        context_split_.append('<PAD>')
                    context_split_ = context_split_[self.n:]

                elif gen_word =='<PAD>' or gen_word =='<PAD>.' or gen_word =='.<PAD>':
                    for i in range(self.n):
                        context_split_.append('<PAD>')
                    context_split_ = context_split_[self.n:]
                    if gen[-2] == '.':
                        continue
                    else:
                        gen = gen[:-1]+ '. '
                else:
                    gen += gen_word + ' '
                    context_split_.append(gen_word)
                    context_split_ = context_split_[1:]
                    count+=1
                old = gen_word
            return gen[:-1]

        def evaluate_text_chars(self,text,spell=False):
            text = normalize_text(text)
            context_split = [text[i:i + self.n] for i in range(len(text) - self.n + 1)]
            count = 0
            for j in context_split:
                r_n = ''
                r_n_1 = ''
                for i in (range(self.n)):
                    r_n += j[i] + '_'
                    if i != self.n-1:
                        r_n_1 += j[i] + '_'
                r_n = r_n[:-1]
                r_n_1 = r_n_1[:-1]
                if self.model_dict[r_n] == 0:
                    count += math.log((self.model_dict[r_n] + 1) / (self.model_dict_n_1[r_n_1] + len(self.model_dict_1)))
                else:
                    count += math.log(self.model_dict[r_n] / self.model_dict_n_1[r_n_1])
            if spell:
                return math.pow(10, count)
            return count


        def evaluate_text(self, text ,spell=False):
            """Returns the log-likelihood of the specified text to be a product of the model.
               Laplace smoothing should be applied if necessary.

               Args:
                   text (str): Text to evaluate.

               Returns:
                   Float. The float should reflect the (log) probability.
            """
            if self.chars:
                return self.evaluate_text_chars(text)
            text = normalize_text(text)
            text = text.replace(". ", ".")
            text_split = text.split('.')
            count = 0
            if len(text.split(' ')) < self.n and spell:
                return self.model_dict_1[text_split[0]]/len(self.model_dict_1)
            for text in text_split:
                text_split_1 = text.split(' ')
                for loc in range(len(text_split_1)):
                    r_n = ''
                    r_n_1 = ''
                    for i in reversed(range(self.n)):
                        if loc - i < 0:
                            r_n += '<PAD>' + '_'
                            r_n_1 += '<PAD>' + '_'
                        else:
                            r_n += text_split_1[loc - i] + '_'
                            if i != 0:
                                r_n_1 += text_split_1[loc - i] + '_'
                    r_n = r_n[:-1]
                    r_n_1 = r_n_1[:-1]
                    if self.model_dict[r_n]  == 0 :
                        count += self.smooth(' '.join(r_n.split('_')))
                    else:
                        try:
                            count += math.log(self.model_dict[r_n] / self.model_dict_n_1[r_n_1])
                        except:
                            count += self.smooth(' '.join(r_n.split('_')))
            if spell:
                return math.pow(10, count)
            return count

        def smooth(self, ngram):
            """Returns the smoothed (Laplace) probability of the specified ngram.

                Args:
                    ngram (str): the ngram to have its probability smoothed

                Returns:
                    float. The smoothed probability.
            """
            text_split = ngram.split(' ')
            r_n=''
            r_n_1 = ''
            for i in range(self.n):
                r_n += text_split[i] + '_'
                if i != self.n-1:
                    r_n_1 += text_split[i] + '_'
            r_n = r_n[:-1]
            r_n_1 = r_n_1[:-1]
            return  math.log((self.model_dict[r_n] + 1) / (self.model_dict_n_1[r_n_1] + len(self.model_dict_1)))


def normalize_text(text):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.

      Args:
        text (str): the text to normalize

      Returns:
        string. the normalized text.
    """
    if text == None or len(text)==0:
        return ''
    else:
        text = text.lower()
        text = text.replace('\n',' ')
        text = re.sub(r'[^\w\s\n.]', '', text)
        text = re.sub(r'\s+', ' ', text)

    return text



def who_am_i():  # this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Matan Leventer', 'id': '208447029', 'email': 'leventem@post.bgu.ac.il'}