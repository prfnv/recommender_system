import pandas as pd
import numpy as np


def popularity(data_action: pd.DataFrame,
               data: pd.DataFrame,
               action: str) -> pd.DataFrame:
    '''
    Признак популярности.
    
    Args:
        data_action: Датафрейм с данными за август.
        data: Датафрейм содержащий таргет.
        action: Действие пользователя, посмотрел товар/добавил в корзину.
    '''
    
    if action == 'to_cart':
        id_count_Series = data_action[data_action['action_type']]['itemid'].value_counts()
    else:
        id_count_Series = data_action[~data_action['action_type']]['itemid'].value_counts()
    df = id_count_Series.rename_axis('jointitemid').reset_index(name= f'{action}_cnt')
    return data.merge(df, on='jointitemid', how="left").fillna(0)


def ctr(data_action: pd.DataFrame,
        data: pd.DataFrame) -> pd.DataFrame:
    '''
    Признак с подсчетом конверсии.
    
    Args:
        data_action: Датафрейм с данными за август.
        data: Датафрейм содержащий таргет.
    '''
    
    new_data = data_action[['itemid', 'action_type']].copy()
    
    viewed_data = new_data[new_data['action_type'] == False]['itemid'].value_counts()
    df1 = viewed_data.rename_axis('jointitemid').reset_index(name='view_cnt')
    
    to_cart_data = new_data[new_data['action_type'] == True]['itemid'].value_counts()
    df2 = to_cart_data.rename_axis('jointitemid').reset_index(name='to_cart_cnt')
    
    df = df1.merge(df2.set_index('jointitemid'), on='jointitemid')
    
    df['ctr'] = df['to_cart_cnt'] / df['view_cnt']
    
    df.drop(['view_cnt', 'to_cart_cnt'], axis=1, inplace=True)
    return data.merge(df, on='jointitemid', how='left').fillna(0)


def date_first_view(data_action: pd.DataFrame,
                    data: pd.DataFrame) -> pd.DataFrame:
    '''
    Признак даты первого просмотра товара.
    
    Args:
        data_action: Датафрейм с данными за август.
        data: Датафрейм содержащий таргет.
    '''
    
    id_item_series = data_action[~data_action['action_type']].groupby(['itemid'], sort=False)['timestamp'].min()
    df = id_item_series.rename_axis('jointitemid').reset_index(name='novelty_cnt')
    return data.merge(df, on='jointitemid', how="left").fillna(0)


def day_avg_popularity(data_action: pd.DataFrame,
                       data: pd.DataFrame,
                       action: str) -> pd.DataFrame:
    '''
    Признак с подсчетом среднего количества добавлений в корзину/просмотров в день.
    
    Args:
        data_action: Датафрейм с данными за август.
        data: Датафрейм содержащий таргет.
        str: Действие пользователя, посмотрел товар/добавил в корзину.
    '''
    
    df = data_action.copy()
    
    if action == 'to_cart':
        count_in_day_series = df[df['action_type']].groupby(['itemid', 'timestamp'])['itemid'].count()
    else:
        count_in_day_series = df[~df['action_type']].groupby(['itemid', 'timestamp'])['itemid'].count()
         
    df_count_in_day = count_in_day_series.rename_axis(['jointitemid', 'timestamp']).reset_index(name='count_day')
    
    df_count_in_day[f'{action}_day_avg_cnt'] = df_count_in_day.groupby(['jointitemid'])['count_day'].transform('mean')
    df_count_in_day.drop(['timestamp', 'count_day'], axis=1, inplace=True)
    df_count_in_day = df_count_in_day.drop_duplicates('jointitemid').reset_index(drop=True)
    
    return data.merge(df_count_in_day, on='jointitemid', how='left').fillna(0)


def views_last_day(data_action: pd.DataFrame, 
                   data: pd.DataFrame) -> pd.DataFrame:
    '''
    Признак c подсчетом количества просмотров в последний день.
    
    Args:
        data_action: Датафрейм с данными за август.
        data: Датафрейм содержащий таргет.
    '''
    
    df = data_action.copy()
    
    data_views = df[~df['action_type']]
    data_last_day_views = data_views[data_views['timestamp'] == max(data_views['timestamp'])]
    df_last_day = data_last_day_views.groupby(['itemid'])['timestamp'].size()
    new_df = df_last_day.rename_axis('jointitemid').reset_index(name='last_day_views_cnt')
    return data.merge(new_df, on='jointitemid', how='left').fillna(0)


def cart_add_last_day(data_action: pd.DataFrame,
                      data: pd.DataFrame) -> pd.DataFrame:
    '''
    Признак c подсчетом количества добавлений в корзину в последний день.
    
    Args:
        data_action: Датафрейм с данными за август.
        data: Датафрейм содержащий таргет.
    '''
    
    df = data_action.copy()
    
    data_tocart = df[df['action_type']]
    data_last_day_tocart = data_tocart[data_tocart['timestamp'] == max(data_tocart['timestamp'])]
    df_last_day = data_last_day_tocart.groupby(['itemid'])['timestamp'].size()
    new_df = df_last_day.rename_axis('jointitemid').reset_index(name='last_day_to_cart_cnt')
    return data.merge(new_df, on='jointitemid', how="left").fillna(0)


def relations(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Создание двух признаков: 
        1. Отношение количества просмотров в последний день
            к среднему количеству просмотров в день.
        2. Отношение количества добавлений в корзину в последний день
            к среднему количеству просмотров в день.
    
    Args:
         data: Датафрейм содержащий таргет.
    '''
    
    data['relation_ldv_mean'] = data['last_day_views_cnt'] / data['view_day_avg_cnt']
    data['relation_ldtocart_mean'] = data['last_day_to_cart_cnt'] / data['to_cart_day_avg_cnt']
    return data.fillna(0)


def daily_views_to_cart(user_actions: pd.DataFrame,
                        action: str) -> pd.DataFrame:
    '''
    Количество просмотров/добавлений в корзину в зависимости от дня недели.
    
    Args:
        data_actions: Датафрейм с данными за август.
        action: Действие пользователя, посмотрел товар/добавил в корзину.
    '''
    
    if action == 'view':
        condition = (
            user_actions['timestamp'] > (user_actions['timestamp'].max() - 7)
        ) & (user_actions['action_type'] == False)
    else:
        condition = (
            user_actions['timestamp'] > (user_actions['timestamp'].max() - 7)
        ) & (user_actions['action_type'] == True)

    # создаем две таблички all_items - все товары, days - номер дня
    all_items = user_actions[['itemid']].drop_duplicates()
    days = pd.DataFrame({'day': [0, 1, 2, 3, 4, 5, 6]})

    # соединим эти таблички, каждому айтему присоединим таблицу с днями

    all_items['key'] = 1
    days['key'] = 1
    item_day_df = all_items.merge(days, on='key').drop(columns=['key'])

    cnts_df = (
        user_actions[condition]
        .groupby(['itemid', 'timestamp'])['clientid']
        .agg(['count'])
        .reset_index()
    )

    cnts_df['day'] = cnts_df['timestamp'] - cnts_df['timestamp'].min()

    item_day_df = (
        item_day_df
        .merge(cnts_df.drop(columns=['timestamp']), on=['itemid', 'day'], how='left')
        .fillna(0)
    )
    item_day_df = item_day_df.rename(columns={'count': 'count_'+action})
    return item_day_df


def get_coef(data: pd.DataFrame, action: str) -> np.float64:
    '''
    Угловой коэффициент прямой, построенной по следующим точкам. 
    Ось x - день недели, ось y - количество добавлений в корзину
    Прямая строится методом МНК. Данная функция считает коэффициент для одного товара.
    
    Args:
        data_actions: Датафрейм с днями.
        action: Действие пользователя, посмотрел товар/добавил в корзину.
    '''
    
    x = data['day'].values 
    y = data['count_'+action].values 
    if (7 * (x**2).sum() - x.sum()**2) == 0:
        return 0
    k = (7 * (x * y).sum() - x.sum()*y.sum()) / (7 * (x**2).sum() - x.sum()**2)
    return k


def mnk_coef(item_day: pd.DataFrame, action: str) -> pd.DataFrame:
    '''
    Метод наименьших квадратов.
    
    Args:
        data_actions: Датафрейм с данными за август.
        action: Действие пользователя, посмотрел товар/добавил в корзину.
    '''
    
    dfs = []
    ks = []
    items = []
    for group_name, group in item_day.groupby('itemid'):
        k = get_coef(group, action)
        items.append(group_name)
        ks.append(k)
    return pd.DataFrame({'itemid': items, 'mnk_'+action: ks})