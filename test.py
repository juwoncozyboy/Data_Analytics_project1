# test.py
# !!! 전부 수정 금지 !!!
# 검증을 수행한다던가 등등의 시도를 하고자 할 경우, validation.py와 같이 별도의 파일을 만들어서 진행해주세요.

# 검증은 현재 지정된 것과 동일하게 2023-10-16 ~ 2023-10-27(10일)로 진행되며,
# 테스트는 제출 전후 영업일 10일 간의 데이터로 진행됩니다. 
# 즉, 테스트 시에는 END_DATE를 '2023-12-01' 근방으로 변경한 후 test.py를 실행할 예정입니다.



import os
from omegaconf import OmegaConf

import pandas as pd
import torch
import dataset as module_data
import model as module_arch
import metric as module_metric
from utils import MetricTracker, get_data_path


END_DATE = '2023-10-27'

    
def get_test_dollar_price(start_date, end_date):
    """
    Do not fix this function
    """
    df = pd.read_csv(get_data_path('USD'), index_col="Date", parse_dates=True, na_values=['nan'])
    df.sort_index(inplace=True)
    price = df.loc[start_date:end_date, ['Price']][-10:].values
    return price


def main(config):
    
    # 데이터 불러오기
    test_dataset = getattr(module_data, config.dataset.type)(end_date=END_DATE, 
                                                             is_training=False, 
                                                             **config.dataset.args)
    # 데이터로더 설정하기
    test_dataloader = getattr(torch.utils.data, config.dataloader.type)(test_dataset, 
                                                                        batch_size=1,
                                                                        shuffle=False,
                                                                        num_workers=0,)

    ###################################################################################################################
    # !!! 수정 금지 !!! : inspect test data
    if abs(test_dataset.y.numpy() - get_test_dollar_price('2023-10-03', END_DATE)).sum() > 1e-3:
        raise ValueError('your test data is wrong!')
    ###################################################################################################################

    # 모델 불러오기
    model = getattr(module_arch, config.model.type)(input_size=test_dataset.__getitem__(0)[0].size(0), **config.model.args)
    model.load_state_dict(torch.load(config.test.load_path))
    model.eval()

    # GPU 사용 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델을 GPU 메모리에 올리기
    model = model.to(device)

    # metric 설정
    metrics = [getattr(module_metric, met) for met in config.test.metrics]
    metric_tracker_test = MetricTracker(*config.test.metrics)

    # 테스트 진행
    for i, (x, y) in enumerate(test_dataloader):
        x, y = x.to(device), y.to(device)
        pred_y = model(x)
        print(f'[DAY {i+1:02d}] predict : {pred_y[0].item():.1f} | target : {y[0].item():.1f}')
        for met in metrics:
            metric_tracker_test.update(met.__name__, met(pred_y, y))
        
    print('\nTEST ' + ',  '.join([f'{k.upper()}: {v:.2f}' for k, v in metric_tracker_test.result().items()]))

    

if __name__ == '__main__':

    src_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.dirname(src_path))  # 프로젝트 폴더로 이동
    
    cfg = OmegaConf.load(os.path.join('src', f'config.yaml'))
    
    main(cfg)
