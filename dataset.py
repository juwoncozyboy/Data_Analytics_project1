# dataset.py
# TODO가 아닌 부분도 얼마든지 수정 가능합니다.
# 단, 수정 금지라고 쓰여있는 항목에 대해서는 수정하지 말아주세요. (불가피하게 수정이 필요할 경우 메일로 미리 문의)

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import get_data_path
import matplotlib.pyplot as plt
import seaborn as sns


SYMBOLS = ['BrentOil', 'Copper', 'CrudeOil', 'Gasoline', 'Gold', 'NaturalGas', 'Platinum', 'Silver',
           'AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD']  # !!! 수정 금지 !!!


class PriceDataset(Dataset):
    def __init__(self, 
                 start_date, 
                 end_date, 
                 is_training=True, 
                 in_columns=['USD_Price','BrentOil_Price', 'Copper_Price', 'CrudeOil_Price', 'Gasoline_Price', 'Gold_Price', 
                             'NaturalGas_Price', 'Platinum_Price', 'Silver_Price', 'AUD_Price', 
                             'CNY_Price', 'EUR_Price', 'GBP_Price', 'HKD_Price', 'JPY_Price'], 
                 out_columns=['USD_Price'],
                 input_days=3, 
                 data_dir='data'):
        excluded_columns = ['CrudeOil_Price','HKD_Price', 'Silver_Price', 'AUD_Price', 'Copper_Price']

        # in_columns가 None이 아닐 때, 제외하고 싶은 열을 제거
        if in_columns is not None:
            in_columns = [col for col in in_columns if col not in excluded_columns]
        self.x, self.y = make_features(start_date, end_date, 
                                       in_columns, out_columns, input_days, 
                                       is_training,  data_dir)
    
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # self.x[idx]의 사이즈는 현재 (input_days, input_dim)이므로, 이를 (input_days * input_dim)으로 flatten함
        return torch.flatten(self.x[idx]), self.y[idx]



# MACD
def calculate_macd(data, short_window, long_window, signal_window):
        short_ema = data.ewm(span=short_window, adjust=False).mean()
        long_ema = data.ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal_line


def calculate_rsi(data, window):
        # RSI 계산
        delta = data.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        rs = up.rolling(window).mean() / down.rolling(window).mean()
        rsi = 100 - 100 / (1 + rs)
        return rsi


# 변동성 지표
def calculate_volatility(data, window):
        return data.rolling(window=window).std()

    

def make_features(start_date, end_date, in_columns, out_columns, input_days, 
                  is_training, data_dir='data'):

    start, end = ''.join(start_date.split('-'))[2:], ''.join(end_date.split('-'))[2:]
    save_fname = f'all_{start}_{end}.pkl'

    if os.path.exists(os.path.join(data_dir, save_fname)):
        print(f'loading from {os.path.join(data_dir, save_fname)}')
        table = pd.read_pickle(os.path.join(data_dir, save_fname))
    
    else:
        print(f'making features from {start_date} to {end_date}')
        table = merge_data(start_date, end_date, symbols=SYMBOLS, data_dir=data_dir)
        table.to_pickle(os.path.join(data_dir, save_fname))
        print(f'saved to {os.path.join(data_dir, save_fname)}')
    
    

    


    # TODO: 데이터 클렌징 및 전처리
    # 주의 : USD_Price에는 값이 있고, 나머지에는 값이 없는 경우가 있음. 이러한 경우는 삭제되지 않도록 주의할 것
    #       만일 삭제될 경우 test.py에서 에러가 발생하여 0점 처리됨
     

    #table.dropna(inplace=True, subset=['USD_Price','Gold_Price', 'Silver_Price'])
    table.dropna(inplace=True, subset=['USD_Price'])
    table.fillna(0, inplace=True)
    #결측치를 0 으로 채우지 않고 선형보간법 사용 
    #table.interpolate(method='linear',inplace=True)
    
    
    # 2020년 데이터를 제거 why) 팬데믹, 미중갈등, 브렉시트
    table = table[~table.index.year.isin([2020])]
    table = table[~table.index.year.isin([2021])]
    table = table[~table.index.year.isin([2022])]
    
    

    

    # 주의 : 미국 환율 가격을 예측해야 하므로, config.yaml의 out_columns에는 반드시 'USD_Price'가 포함되어야 함
    if 'USD_Price' not in out_columns:
        raise ValueError('USD_Price must be included in out_columns')   # !!! 수정 금지 !!!
    
    use_columns = list(set(in_columns + out_columns))  # 중복 제거
    df = table[use_columns]
    
   
    


    

    


    # TODO: 추가적인 feature engineering이 필요하다면 아래에 작성
    # 가령, 주식 데이터의 경우 이동평균선, MACD, RSI 등의 feature를 생성할 수 있음
    # 주의 : 미래 데이터를 활용하는 일이 없도록 유의할 것 (가령, 10월 31일 데이터(row)에 10월 31일 뒤의 데이터가 활용되면 안 됨)
    # 주의 : 추가로 활용할 feature들은 in_columns에도 추가할 것
    in_columns += []
    df['MACD'], df['Signal_Line'] = calculate_macd(df['USD_Price'], short_window=12, long_window=26, signal_window=9)
    in_columns.extend(['MACD', 'Signal_Line'])

    df['RSI'] = calculate_rsi(df['USD_Price'], window=14)
    df['RSI'].fillna(0, inplace=True)
    
    in_columns.append('RSI')

    df['Volatility'] = calculate_volatility(df['USD_Price'], window=20)
    df['Volatility'].fillna(0, inplace=True)
    in_columns.append('Volatility')

    
    
    # 5일이평선 
    #df['USD_Price_MA5'] = df['USD_Price'].rolling(window=5).mean()
    #df['USD_Price_MA5'] = df['USD_Price'].rolling(window=5).mean().fillna(method='ffill')
        # 먼저 이동 평균선 계산
    #df['USD_Price_MA5'] = df['USD_Price'].rolling(window=5).mean()


    
    
    


    #df 출력해봄
    print(df.shape)
    print(df.head())
    
    # 결측치 확인
    missing_values = df.isnull().sum()
    print(missing_values)

    # 전체 데이터에서 결측치가 있는 컬럼의 수를 확인
    total_missing_columns = missing_values[missing_values > 0].count()
    print(f"Number of columns with missing values: {total_missing_columns}")

    # 결측치가 있는 행의 수를 확인
    total_missing_rows = df.isnull().any(axis=1).sum()
    print(f"Number of rows with missing values: {total_missing_rows}")
    
    
    

    #상관관계
    print(df.corr())

    #상관관계 시각화
    plt.figure(figsize=(12, 8))  # 히트맵의 크기 설정
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
    plt.show()
    #plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(14, 7))
    for col in in_columns:
        if 'Price' in col:
            plt.plot(df[col], label=col)
    plt.title('가격 시계열')
    plt.xlabel('시간')
    plt.ylabel('가격')
    plt.legend()
    plt.show()

    # 시각화 - MACD와 Signal Line
    plt.figure(figsize=(14, 7))
    plt.plot(df['MACD'], label='MACD')
    plt.plot(df['Signal_Line'], label='Signal Line')
    plt.title('MACD와 Signal Line')
    plt.xlabel('시간')
    plt.ylabel('값')
    plt.legend()
    plt.show()

    # 시각화 - RSI
    plt.figure(figsize=(14, 7))
    plt.plot(df['RSI'], label='RSI')
    plt.title('상대강도지수 (RSI)')
    plt.xlabel('시간')
    plt.ylabel('RSI 값')
    plt.legend()
    plt.show()

    # 시각화 - 변동성
    plt.figure(figsize=(14, 7))
    plt.plot(df['Volatility'], label='Volatility')
    plt.title('변동성')
    plt.xlabel('시간')
    plt.ylabel('변동성 값')
    plt.legend()
    plt.show()

    

    # 5일 이동평균선과 USD_Price가 높은 상관관계를 보이므로 5일 이평선 제거
    # MACD와 Signal_Line간의 상관관계가 높으므로 Signal_Line만 가짐

    #for column in df.columns:
    #    plt.figure(figsize=(10, 4))
    #    sns.histplot(df[column], kde=True)
    #    plt.title(f'Distribution of {column}')
    #    plt.show()

    # First, ensure that the index is a datetime type
    #df.index = pd.to_datetime(df.index)

    # Now, group by the year and calculate the mean for each year
    

    # Plotting
    #plt.figure(figsize=(14, 7))
    #sns.lineplot(data=df.columns)  # This plots all columns by default
    #plt.title('Yearly Averages of the Features')
    #plt.xlabel('Year')
    #plt.ylabel('Average Value')
    #plt.show()




    # input_days 만큼의 과거 데이터를 사용하여 다음날의 USD_Price를 예측하도록 데이터셋 구성됨
    date_indices = sorted(table.index)
    #print(date_indices)
    x = np.asarray([df.loc[date_indices[i:i + input_days], in_columns] for i in range(len(df) - input_days)])
    y = np.asarray([df.loc[date_indices[i + input_days], out_columns] for i in range(len(df) - input_days)])


    # 최근 10일을 test set으로 사용
    # 주의 : 검증 및 테스트 과정에 반드시 최근 10일 데이터를 사용해야 하므로 수정하지 말 것
    training_x, test_x = x[:-10], x[-10:]  # !!! 수정 금지 !!!
    training_y, test_y = y[:-10], y[-10:]  # !!! 수정 금지 !!!

    
    return (training_x, training_y) if is_training else (test_x, test_y)



def merge_data(start_date, end_date, symbols, data_dir='data'):

    dates = pd.date_range(start_date, end_date, freq='D')
    df = pd.DataFrame(index=dates)

    if 'USD' not in symbols:
        symbols.insert(0, 'USD')

    for symbol in symbols:
        df_temp = pd.read_csv(get_data_path(symbol, data_dir), index_col="Date", parse_dates=True, na_values=['nan'])
        df_temp = df_temp.reindex(dates)
        df_temp.columns = [symbol + '_' + col for col in df_temp.columns]  # rename columns
        df = df.join(df_temp)

    return df




if __name__ == "__main__":

    start_date = '2013-01-01'
    end_date = '2023-10-27'
    is_training = False

    test_data = PriceDataset(start_date, end_date, 
                             is_training=is_training,
                             in_columns=['BrentOil_Price', 'Copper_Price', 'CrudeOil_Price', 'Gasoline_Price', 'Gold_Price', 'NaturalGas_Price', 'Platinum_Price', 'Silver_Price', 'AUD_Price', 'CNY_Price', 'EUR_Price', 'GBP_Price', 'HKD_Price', 'JPY_Price', 'USD_Price'],
                             out_columns=['USD_Price'],                              
                             input_days=5,
                             data_dir='data')
     # USD_Price 텐서 불러오기
    usd_prices = [test_data[i][1] for i in range(len(test_data))]  # [1]은 y값, 즉 USD_Price를 가져옴
    usd_prices_tensor = torch.stack(usd_prices)
    #print(in_columns)
    # 텐서 출력
    print(usd_prices_tensor)
    print(f'\ntest_data length : {len(test_data)}')
    print(f'\ndataset_x_original[9] : \n{test_data.x[9]}')
    print(f'\ndataset_x_flatten[9] : \n{test_data.__getitem__(9)[0]}')
    print(f'\ndataset_y[9] : \n{test_data.__getitem__(9)[1]}')

    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=0)
    print(f'\ntest_dataloader length : {len(test_dataloader)}')
