import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize


def calculate_session_similarity(data: pd.DataFrame,
                                 matrix: sp.csr_matrix, 
                                 action: str) -> pd.DataFrame:
    """
    Функция для подсчета:
        Схожести товаров от частоты попадания в одну сессию;
        Схожести товаров от количества сессий, в которых встречались товары.
        
    Args:
        data Датафрейм с признаками:
        matrix: Матрица товар/сессия.
        action: Действие пользователя, посмотрел товар/добавил в корзину.
    """
    
    pairs = data.copy()
    
    pairs = pairs.dropna()[['item_cat', 'jointitem_cat']]
    pairs['item_cat'] = pairs['item_cat'].astype('uint32')
    pairs['jointitem_cat'] = pairs['jointitem_cat'].astype('uint32')

    pairs['same_items_on_session_'+action] = (
      normalize(matrix[pairs['item_cat'].values], axis=1)
      .multiply(normalize(matrix[ pairs['jointitem_cat'].values], axis=1))
      .sum(axis=1)
    )
    
    pairs['count_on_session_'+action] = (
      matrix[pairs['item_cat'].values]
      .multiply(matrix[ pairs['jointitem_cat'].values])
      .sum(axis=1)
    )

    data = (
        data
        .merge(pairs, on=['item_cat', 'jointitem_cat'], how='left')
        .drop_duplicates()
        .reset_index(drop=True)
    )
    
    return data