import pandas as pd
import numpy as np


def reindexing_sessionid_itemid(session_item: pd.DataFrame) -> pd.DataFrame:
    '''
    Функция для переиндиксации индеков сессий и товаров.
    
    Args:
        session_item: Датафрейм с пользовательскими сессиями.
    '''
    session = session_item['sessionid'].unique() # все уникальные айди клиентов из user_actions
    session_cat = np.arange(0, len(session), dtype='uint32') # массив с элементами от 0 до n - 1

    # таблица, где каждая строка переводит clientid в номер строки
    user_mapping = pd.DataFrame({'old': session, 'new': session_cat})

    items = session_item['itemid'].unique() # все уникальные айди товаров из user_actions
    items_cat = np.arange(0, len(items), dtype='uint32') # массив с элементами от 0 до m - 1

    # таблица, где каждая строка переводит itemid в номер столбца
    item_mapping = pd.DataFrame({'old': items, 'new': items_cat})

    session_item['sessionid'] = session_item['sessionid'].map(user_mapping.set_index('old')['new'])
    session_item['itemid'] = session_item['itemid'].map(item_mapping.set_index('old')['new'])
    return (session_item, user_mapping, item_mapping)


def reindexing_clientid_itemid(user_actions: pd.DataFrame) -> pd.DataFrame:
    '''
    Функция для переиндиксации индеков клиентов и товаров.
    
    Args:
        user_action: Датафрейм с данными за август.
    '''
    
    clients = user_actions['clientid'].unique() # все уникальные айди клиентов из user_actions
    clients_cat = np.arange(0, len(clients), dtype='uint32') # массив с элементами от 0 до n-1

    # таблица, где каждая строка переводит clientid в номер строки
    user_mapping = pd.DataFrame({'old': clients, 'new': clients_cat})

    items = user_actions['itemid'].unique() # все уникальные айди товаров из user_actions
    items_cat = np.arange(0, len(items), dtype='uint32') # массив с элементами от 0 до m-1

    # таблица, где каждая строка переводит itemid в номер столбца
    item_mapping = pd.DataFrame({'old': items, 'new': items_cat})

    user_actions['clientid'] = user_actions['clientid'].map(user_mapping.set_index('old')['new'])
    user_actions['itemid'] = user_actions['itemid'].map(item_mapping.set_index('old')['new'])
    return (user_actions, user_mapping, item_mapping)


def reindexing_itemid(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция для переиндексации индексов товаров.
    
    Args:
        data: Датафрейм с текстовым описанием товаров.
    """
    
    items = data['itemid'].unique()
    items_cat = np.arange(0, len(items), dtype='uint32')
    
    items_mapping = pd.DataFrame({'old': items, 'new': items_cat})
    
    data['itemid'] = data['itemid'].map(items_mapping.set_index('old')['new'])
    
    return (data, items_mapping)