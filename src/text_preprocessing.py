import pandas as pd
from pymystem3 import Mystem


def text_preprocessing(text_col: pd.Series, stopwords: list) -> pd.Series:
    """
    Функция для предобработки названий и текстового описания товара.
    
    Args:
        text_col: Столбец с текстовой информацией.
        stopwords: Список стоп-слов.
    """
    
    lemmatize_func = Mystem().lemmatize
    
    pattern = r'\b(?:{})\b'.format('|'.join(stopwords))
    text = (
        text_col
        .str.lower()
        .str.replace(r'<[^>]+>|[^a-zа-яё0-9]', ' ')
        .str.replace(r'(\s)', ' ')
        .str.strip() # удаление пробелов в начале и в конце
#         .apply(lambda x: lemmatize_func(x) if isinstance(x, str) else None) # лемматизация
#         .apply(lambda x: ' '.join(x) if isinstance(x, list) else None) # соединение лемматизированных слов
        .str.replace(pattern, ' ') # удаление стоп-слов
        .str.replace(r'\b(\w)\b', '') # удаление слов из одной буквы
        .str.replace(r'\s+', ' ') # любое количество пробелов на 1 пробел
        .str.strip() # удаление пробелов в начале и в конце
    )
    
    return text