# библиотеки для работы с нейросетями
import re
import torch 
from transformers import AutoTokenizer, AutoModel

# работа со строками
import string

# косинусные расстояния
from sklearn.metrics.pairwise import cosine_similarity

class IntentRecognizer():
    # загружаем предварительно натренированную нейросеть и токенизатор
    def __init__(self, cut_length = 30):
        self.cut_length = cut_length # длина обрезки эмбеддингов (подбирается эмпирически)
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        self.model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
   
    # вычисляет нормализованное скалярное произведение для сравнения 
    def cosine_sim(self,a, b):
        return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))

    # убирает пунктуацию, так повышается точность
    def remove_punctuation(self,text):
        return "".join([ch if ch not in string.punctuation + '«' + '»' + '—' else ' ' for ch in text])
    
    # Функция автора модели, нужная для извлечения эмбеддингов (векторных представлений текста)
    def embed_bert_cls(self,text, model, tokenizer): 
        # убираем пунктуацию и переводим в нижний регистр
        text = self.remove_punctuation(text).lower()
        
        t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            model_output = model(**{k: v.to(model.device) for k, v in t.items()})
        
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings[0].cpu().numpy()
        
    # возвращает эмбеддинги
    def get_point(self,sentense):
        # Эмбеддинги обрезаются до длины cut_length 
        return self.embed_bert_cls(sentense, self.model, self.tokenizer)[0:self.cut_length]
    
    def one_world(self,questions,intents):
        '''Оставляет однословные вопросы и соответствующие им интенты'''
        ow_questions = []
        ow_intents = []
        for i,q in enumerate(questions):
            if len(q.split()) == 1:
                # сохраняем шаблоны для поиска
                ow_questions.append(re.compile(f'\w*{q.lower()}\w*'))
                ow_intents.append(intents[i])
        return ow_questions,ow_intents

    def load_data(self,questions,intents):
        '''Для начала работы необходимо загрузить кортежи (или списки) вопросов и
        соответствующих им намерений, далее вопросы преобразуются в эмбеддинги для сравнения'''    
        self.embeddings = [self.get_point(q) for q in questions]
        self.intents = intents
        self.ow_questions, self.ow_intents = self.one_world(questions,intents)
    
    def preprocess_text(self,question,minlen) -> list:
        '''Удаляет спецсимволы и делит текст на части по знакам препинания'''
        # заменяем спецсимволы пробелами
        text = re.sub("[«»—\"#$%&'()\*\+\-/:<=>@[\\]^_`{|}~\\\\]",' ',question)
        # если идёт несколько пробелов подряд, заменяем на один
        text = re.sub("\s+",' ',text)
        # делим текст по знакампрепинания
        text = re.split("[!,.;?]",text)
        # отрезаем лишние пробелы и убираем короткие тексты
        text_list = [q.strip() for q in text if len(q.strip()) >= minlen]
        return text_list
    
    def make_intent_list(self,text_list) -> list:
        '''Выставляет каждому элементу в списке интент и схожесть'''
        intent_list = []
        for text in text_list: 
            if len(text.split()) ==1:#Если всего одно слово - ищем через regex
                intent_sim = ['ignore', 0.801]# по умолчанию игнор
                for i,q in enumerate(self.ow_questions):
                    result = q.search(text_list[0].lower())
                    if result != None:# если есть частичное совпадение 
                        intent_sim = [self.ow_intents[i], 0.801]
                intent_list.append(intent_sim)        
            
            else:        
                similarity = 0
                closest = 0 
                # преобразуем вопрос в эмбеддинг
                q_emb  = self.get_point(text)
                # идём по списку с эмбеддингами 
                for i in range(len(self.embeddings)):
                    # вычисляем схожесть
                    act_dist = self.cosine_sim(self.embeddings[i] , q_emb)[0][0]
                    # сравниваем с минимальной
                    if act_dist > similarity:
                        similarity = act_dist
                        # получем номер самого похожего вопроса
                        closest = i
                # выводим намерение, соответствующее вопросу и схожесть
                intent_list.append([self.intents[closest],similarity]) 
        return intent_list           
    
    def get_suitable_intent(self,intent_list) -> tuple:
        '''Выводит из списка интент с максимальной схожестью. 
        При этом игнорирует приветствие, чтобы выявить суть'''

        if len(intent_list) > 1:
                # убираем приветствие
                intent_list = [k for k in intent_list if 'hello' not in k ]
                if len(intent_list) > 0:
                    # находим максимальнуюю схожесть и её интент
                    intent = max(intent_list,key=lambda x: x[1])[0]
                    similarity= max(intent_list,key=lambda x: x[1])[1]
                    return intent,similarity
                else:
                    # если список был пустой, значит в нём было только приветствие
                    return 'hello',1
        else:
            # если в списке был всего один интент, выводим его
            return intent_list[0][0] ,intent_list[0][1] 

    def get_intent(self,question,minlen = 5) -> tuple:
        '''Функция получает вопрос в виде текста (question), преобразует его в эмбеддинг.
        Затем этот вопрос сравнивается со списком загруженных вопросов и по номеру самого близкого
        возвращает соответствующее намерение (intent) и схожесть (similarity).Если длина вопроса 
        меньше minlen, то выводится ignore '''
                     
        text_list = self.preprocess_text(question,minlen)
        # Игнорируем короткие 
        if len(text_list) == 0:
            return 'ignore',1
                
        else:    
            intent_list = self.make_intent_list(text_list)
            return self.get_suitable_intent(intent_list)      

'''---------------- Как использовать --------------------'''

'''Импортируем ...'''
#from intent_recognizer import *

if __name__ == "__main__":
    # Примеры кортежей с вопросами и намерениями 
    questions = ('Привет', 'Здравствуйте', 'Пока',"До свидания")
    intents = ('hello','hello','bye','bye')
    # создаём объект класса 
    recognizer = IntentRecognizer()
    # загружаем данные
    recognizer.load_data(questions,intents)
    # находим интент и схожесть
    intent, similarity = recognizer.get_intent('Приветик')
    print(intent,'\n',similarity) 
    
